# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import logging
import os
import random
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.special import softmax
from torch.nn import BCEWithLogitsLoss
from multilabel_evaluation import evaluate as evaluate_multilabel
from model import * 
import gc
# del variables
gc.collect()
torch.cuda.empty_cache()

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	AlbertConfig,
	AlbertForSequenceClassification,
	AlbertTokenizer,
	BertConfig,
	BertForSequenceClassification,
	BertTokenizer,
	DistilBertConfig,
	DistilBertForSequenceClassification,
	DistilBertTokenizer,
	RobertaModel,
	RobertaConfig,
	RobertaForSequenceClassification,
	RobertaTokenizer,
	XLMConfig,
	XLMForSequenceClassification,
	XLMRobertaConfig,
	XLMRobertaForSequenceClassification,
	XLMRobertaTokenizer,
	XLMTokenizer,
	XLNetConfig,
	XLNetForSequenceClassification,
	XLNetTokenizer,
	get_linear_schedule_with_warmup,
	AutoModel, 
	AutoTokenizer,
)

#from transformers import glue_compute_metrics as compute_metrics
#from transformers import glue_convert_examples_to_features as convert_examples_to_features
#from transformers import glue_output_modes as output_modes
#from transformers import glue_processors as processors
from citances_processor import citances_processors as processors
from citances_processor import citances_output_modes as output_modes
from citances_processor import citances_convert_examples_to_features as convert_examples_to_features


try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
	"bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
	"xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
	"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
	"roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
	"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
	"albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
	"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

def compute_metrics(preds, labels):
	acc = (preds == labels).mean()
	f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
	p = precision_score(y_true=labels, y_pred=preds, average="macro")
	r = recall_score(y_true=labels, y_pred=preds, average="macro")

	return {
		"acc": acc,
		"f1": f1,
		"acc_and_f1": (acc + f1) / 2,
		"p":p,
		"r":r
	}

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def load_triple(dataset):
	import json 
	path  = "../../datasets/data_multicite_cv/"+dataset+".json"
	out = []
	with open(path , 'r') as f_r:
		full_data = json.load(f_r)
		for data in full_data:
			out.append(data['list_cfu'])
	return out 

# def load_word_graph(dataset):
# 	import json 
# 	path  = "data_multicite_triple_with_graph/"+dataset+".json"
# 	out = []
# 	with open(path , 'r') as f_r:
# 		full_data = json.load(f_r)
# 		for data in full_data:
# 			# out.append([data['map_tok2tok'] , data['map_triple2tok'] , data['map_tok2triple'] ] )
# 			out.append([ [] , data['map_triple2tok'] , data['map_tok2triple'] ] )

# 	return out 	


def train(args, features, model, tokenizer, processor):

	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	print('number training  ' , len(features) )
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(features) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(features) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# Check if saved optimizer or scheduler states exist
	if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
		os.path.join(args.model_name_or_path, "scheduler.pt")
	):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
		)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(features))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	# Check if continuing training from a checkpoint
	if os.path.exists(args.model_name_or_path):
		# set global_step to gobal_step of last saved checkpoint from model path
		global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])

		logger.info("  Continuing training from checkpoint, will skip to saved global_step")
		logger.info("  Continuing training from epoch %d", epochs_trained)
		logger.info("  Continuing training from global step %d", global_step)
		logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	# train_iterator = trange(
	#     epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
	# )
	set_seed(args)  # Added here for reproductibility

	bestS , bestW = 0. , 0. 
	import time 
	save_output =  str(time.time())
	print('save output ', save_output)
	os.makedirs(save_output)
	import json
	args.device = 'cuda'
	print(vars(args) , type(vars(args)))
	with open(save_output + '/setting.json' , 'w') as f_setting:
		json.dump( vars(args), f_setting) 

	train_triple = load_triple('train')
	# train_word_graph = load_word_graph('train')
	print('Finish load triple information ')


	for epoch in range(int(args.num_train_epochs)):
		# epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		original_index = list(range(len(features)))
		np.random.shuffle(original_index)
		list_index_batch = [original_index[x:x+args.train_batch_size] for x in range(0, len(original_index), args.train_batch_size)]
		for step, batch_index in enumerate(list_index_batch):
			global_step += 1
			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue

			model.train()
			# batch = tuple(t.to(args.device) for t in batch)

			if args.classification_type == "multiclass":
				pass 
			elif args.classification_type == "multilabel":
				sample = [ features[t] for t in batch_index ] 
				triple_indexes = [train_triple[t] for t in batch_index]
				chunks_idx , label_binary = [s[0] for s in sample] , [s[2] for s in sample]
				logits , _, _= model(chunks_idx , triple_indexes, tokenizer)
				loss_func = BCEWithLogitsLoss()
				label_binary = torch.stack(label_binary , dim = 0)
				loss = loss_func(logits, label_binary.type_as(logits))

				# logits , _, _= model(chunks_idx , triple_indexes)
				# loss_func = nn.BCELoss()
				# loss = loss_func(logits, label_binary.type_as(logits))
			else:
				raise NotImplementedError()

			if args.n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu parallel training

			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()

			if args.logging_steps > 0 and global_step % args.logging_steps == 0:
				logs = {}

				# gc.collect()
				# torch.cuda.empty_cache()
				# # Khởi động lại bộ đếm đỉnh VRAM về 0 trước khi gọi evaluate
				# torch.cuda.reset_peak_memory_stats()
				with torch.no_grad():
					start = time.time()
					results , result_macro = evaluate(args, model, tokenizer, processor = processor , save_output= save_output, global_step= global_step)
					end = time.time()
					print('Time processing ' , (end-start) / 3313)
				print('global step ' ,	 global_step ,'results ', results)

				# if torch.cuda.is_available():
				# 	inf_peak_bytes = torch.cuda.max_memory_allocated()
				# 	inf_peak_mb = inf_peak_bytes / (1024 ** 2)
				# 	print(f'==> Peak GPU Memory during Evaluation (Process-only): {inf_peak_mb:.2f} MB')

				print('result macro ' , result_macro)

				if results['strict']  >= bestS :
					bestS = results['strict']
					# bestW = results['weak']

					print(bestS , bestW)
					# Save model checkpoint
					# print('result 1', results_1['strict'] , results_1['weak'])
					if args.save_model == 1:
						file = open(save_output + '/checkpoint_best' , 'wb')
						import pickle
						pickle.dump(model , file)
						file.close()
					with open(save_output +'/global_step_best.txt','w') as f:
						f.write(str(global_step))

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break

		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	# if args.local_rank in [-1, 0]:
	#     tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", processor=None, save_output = '.', global_step = 0 ):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
	eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

	results = {}
	for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
		if args.do_eval:
			features = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, test=False)
		if args.do_test:
			features = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, test=True)
		if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(eval_output_dir)
		# Note that DistributedSampler samples randomly

		# multi-gpu eval
		if args.n_gpu > 1:
			model = torch.nn.DataParallel(model)

		# Eval!
		logger.info("***** Running evaluation {} *****".format(prefix))
		logger.info("  Num examples = %d", len(features))
		logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None

		original_index = list(range(len(features)))
		list_index_batch = [original_index[x:x+args.per_gpu_eval_batch_size] for x in range(0, len(original_index), args.per_gpu_eval_batch_size)]
		print('len list index batch ' , len(list_index_batch))
		preds , out_label_ids = [] , [] 

		triple_test = load_triple('test')
		# word_graph_test = load_word_graph('test')
		with open(save_output+'/triple_weight_'+str(global_step)+'.txt' ,'w') as f , open(save_output+'/triple_intent_att_'+str(global_step)+'.txt' ,'w') as f1:
			for  step,  batch_index  in enumerate(list_index_batch):
				# print('step ' , step)
				model.eval()
				sample = [features[t] for t in batch_index]
				triple_indexes = [triple_test[t] for t in batch_index]
				# data_graph = [word_graph_test[t] for t in batch_index]

				chunks_idx , label_binary = [s[0] for s in sample] , [s[2] for s in sample]
				with torch.no_grad():
					if args.classification_type == "multiclass":
						pass
						
					elif args.classification_type == "multilabel":
						logits, triple_weight, triple_intent_att = model(chunks_idx , triple_indexes,  tokenizer)
						for j in range(len(triple_weight)):
							f.write(" ".join([str(t) for t in triple_weight[j]]) + "\n")
							f1.write('<fff>'.join(' '.join(f'{x:.3f}' for x in row) for row in triple_intent_att[j]) + "\n")
						loss_func = BCEWithLogitsLoss()
						label_binary = torch.stack(label_binary, dim = 0)
						tmp_eval_loss = loss_func(logits, label_binary.type_as(logits))

					eval_loss += tmp_eval_loss.mean().item()
				nb_eval_steps += 1
	
				# preds.append(logits.detach().cpu().numpy().tolist())
				# out_label_ids.append(sample[-1].detach().cpu().numpy().tolist())
				preds.extend(logits.detach().cpu().numpy().tolist())
				out_label_ids.extend(label_binary.detach().cpu().numpy().tolist())
					# print('out put da them', out_label_ids)
		preds = np.asarray(preds)
		out_label_ids = np.asarray(out_label_ids)

		#write the logit to file to check
		eval_loss = eval_loss / len(features)
		# print('eval loss ' , eval_loss)
		pred_probs = None
		if args.output_mode == "classification":
			preds_old = preds
			preds = np.argmax(preds_old, axis=1)
			pred_probs = np.max(softmax(preds_old, axis=1), axis=1)
		if args.classification_type == "multilabel":
			preds_old = preds
			pred_probs = torch.sigmoid(torch.tensor(preds_old)).detach().cpu().numpy()
			# pred_probs = torch.tensor(preds_old).detach().cpu().numpy()

			# TODO: Output all labels with prob > 0.5
			# preds = [np.argwhere(x > 0.5) for x in pred_probs]
			preds = [] 
			for x in pred_probs:
				predict_index = np.argwhere(x > 0.5)
				if predict_index.size == 0:          # không có phần tử nào > 0.5
					i = np.argmax(x)
					predict_index = np.array([[i]]) # format giống argwhere cho vector 1D
				preds.append(predict_index)
		elif args.output_mode == "regression":
			preds = np.squeeze(preds)
		label_list = processor.get_labels()
		if not (args.task_name in ["pride"] and args.do_test):
			if args.classification_type == "multilabel":
				out_label_ids = [np.argwhere(x == 1) for x in out_label_ids]
				preds = [[j[0] for j in l] for l in preds]
				out_label_ids = [[j[0] for j in l] for l in out_label_ids]

				# print('preds ' , preds)
				# print('our labels ' , out_label_ids)
				result = evaluate_multilabel(preds, out_label_ids, label_list)
				# from multilabel_evaluation import compute_macro_for_multicite
				# result_macro = compute_macro_for_multicite(preds, out_label_ids)
				result_macro = 0 

			else:
				result = compute_metrics(preds, out_label_ids)

		if args.do_eval and not args.do_test:
			# print('-------')
			results.update(result)
			output_eval_file = os.path.join(save_output, prefix, "eval_results.txt")
		else:
			# print('-------------------------------------')
			output_eval_file = os.path.join(save_output, prefix, "test_results"+str(global_step)+".txt")
			output_predictions_file = os.path.join(save_output, prefix, "predictions"+str(global_step)+".txt")
			output_predictions_probs_file = os.path.join(save_output, prefix, "predictions_probs"+str(global_step)+".txt")

			with open(output_predictions_file, "w") as writer:
				for pred in preds:
					if args.task_name not in ["kylel"]:
						if args.classification_type == "multilabel":
							writer.write("%s\n" % " ".join([str(label_list[p]) for p in pred]))
						else:
							writer.write("%s\n" % str(label_list[pred]))
					else:
						writer.write("%s\n" % str(pred))

			# with open(output_predictions_probs_file, "w") as writer:
			# 	# for i,pred in enumerate(pred_probs):
			# 	#     writer.write("%s\t%s\n" % (str(preds[i]), str(pred)))
			# 	for i,pred in enumerate(preds_old):
			# 		writer.write("%s\n" % (   str(pred)  ))

			# with open(output_predictions_probs_file, "w") as writer:
			# 	# for i,pred in enumerate(pred_probs):
			# 	#     writer.write("%s\t%s\n" % (str(preds[i]), str(pred)))
			# 	for i,pred in enumerate(preds_old):
			# 		writer.write( " ".join([str(t) for t in pred]) +"\n")


		#comment ở đây
		if not (args.task_name in ["pride"] and args.do_test):
			with open(output_eval_file, "w") as writer:
				logger.info("***** Eval results {} *****".format(prefix))
				for key in sorted(result.keys()):
					logger.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))

	return result, result_macro

def evaluate_1(args, model, tokenizer, prefix="", processor=None, save_output = '.', global_step = 0 ):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
	eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

	results = {}
	for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
		if args.do_eval:
			features = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, test=False)
		if args.do_test:
			features = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, test=True)
		if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(eval_output_dir)
		# Note that DistributedSampler samples randomly

		# multi-gpu eval
		if args.n_gpu > 1:
			model = torch.nn.DataParallel(model)

		# Eval!
		logger.info("***** Running evaluation {} *****".format(prefix))
		logger.info("  Num examples = %d", len(features))
		logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None

		original_index = list(range(len(features)))
		list_index_batch = [original_index[x:x+args.per_gpu_eval_batch_size] for x in range(0, len(original_index), args.per_gpu_eval_batch_size)]
		print('len list index batch ' , len(list_index_batch))
		preds , out_label_ids = [] , [] 

		triple_test = load_triple('test')
		# word_graph_test = load_word_graph('test')
		with open(save_output+'/triple_weight.txt' ,'w') as f , open(save_output+'/triple_intent_att.txt' ,'w') as f1:
			for  step,  batch_index  in enumerate(list_index_batch):
				# print('step ' , step)
				model.eval()
				sample = [features[t] for t in batch_index]
				triple_indexes = [triple_test[t] for t in batch_index]
				# data_graph = [word_graph_test[t] for t in batch_index]

				chunks_idx , label_binary = [s[0] for s in sample] , [s[2] for s in sample]
				with torch.no_grad():
					if args.classification_type == "multiclass":
						pass
						
					elif args.classification_type == "multilabel":
						logits, triple_weight, triple_intent_att = model(chunks_idx , triple_indexes,  tokenizer)
						for j in range(len(triple_weight)):
							f.write(" ".join([str(t) for t in triple_weight[j]]) + "\n")
							f1.write('<fff>'.join(' '.join(f'{x:.3f}' for x in row) for row in triple_intent_att[j]) + "\n")
						loss_func = BCEWithLogitsLoss()
						label_binary = torch.stack(label_binary, dim = 0)
						tmp_eval_loss = loss_func(logits, label_binary.type_as(logits))

					eval_loss += tmp_eval_loss.mean().item()
				nb_eval_steps += 1
	
				# preds.append(logits.detach().cpu().numpy().tolist())
				# out_label_ids.append(sample[-1].detach().cpu().numpy().tolist())
				preds.extend(logits.detach().cpu().numpy().tolist())
				out_label_ids.extend(label_binary.detach().cpu().numpy().tolist())
					# print('out put da them', out_label_ids)
		preds = np.asarray(preds)
		out_label_ids = np.asarray(out_label_ids)

		#write the logit to file to check
		eval_loss = eval_loss / len(features)
		# print('eval loss ' , eval_loss)
		pred_probs = None
		if args.output_mode == "classification":
			preds_old = preds
			preds = np.argmax(preds_old, axis=1)
			pred_probs = np.max(softmax(preds_old, axis=1), axis=1)
		if args.classification_type == "multilabel":
			preds_old = preds
			pred_probs = torch.sigmoid(torch.tensor(preds_old)).detach().cpu().numpy()
			# pred_probs = torch.tensor(preds_old).detach().cpu().numpy()

			# TODO: Output all labels with prob > 0.5
			# preds = [np.argwhere(x > 0.5) for x in pred_probs]
			preds = [] 
			for x in pred_probs:
				predict_index = np.argwhere(x > 0.5)
				if predict_index.size == 0:          # không có phần tử nào > 0.5
					i = np.argmax(x)
					predict_index = np.array([[i]]) # format giống argwhere cho vector 1D
				preds.append(predict_index)
		elif args.output_mode == "regression":
			preds = np.squeeze(preds)
		label_list = processor.get_labels()
		if not (args.task_name in ["pride"] and args.do_test):
			if args.classification_type == "multilabel":
				out_label_ids = [np.argwhere(x == 1) for x in out_label_ids]
				preds = [[j[0] for j in l] for l in preds]
				out_label_ids = [[j[0] for j in l] for l in out_label_ids]

				# print('preds ' , preds)
				# print('our labels ' , out_label_ids)
				result = evaluate_multilabel(preds, out_label_ids, label_list)
				# from multilabel_evaluation import compute_macro_for_multicite
				# result_macro = compute_macro_for_multicite(preds, out_label_ids)
				result_macro = 0 

			else:
				result = compute_metrics(preds, out_label_ids)

		if args.do_eval and not args.do_test:
			# print('-------')
			results.update(result)
			output_eval_file = os.path.join(save_output, prefix, "eval_results.txt")
		else:
			# print('-------------------------------------')
			output_eval_file = os.path.join(save_output, prefix, "test_results"+str(global_step)+".txt")
			output_predictions_file = os.path.join(save_output, prefix, "predictions"+str(global_step)+".txt")
			output_predictions_probs_file = os.path.join(save_output, prefix, "predictions_probs"+str(global_step)+".txt")

			with open(output_predictions_file, "w") as writer:
				for pred in preds:
					if args.task_name not in ["kylel"]:
						if args.classification_type == "multilabel":
							writer.write("%s\n" % " ".join([str(label_list[p]) for p in pred]))
						else:
							writer.write("%s\n" % str(label_list[pred]))
					else:
						writer.write("%s\n" % str(pred))

			# with open(output_predictions_probs_file, "w") as writer:
			# 	# for i,pred in enumerate(pred_probs):
			# 	#     writer.write("%s\t%s\n" % (str(preds[i]), str(pred)))
			# 	for i,pred in enumerate(preds_old):
			# 		writer.write("%s\n" % (   str(pred)  ))

			# with open(output_predictions_probs_file, "w") as writer:
			# 	# for i,pred in enumerate(pred_probs):
			# 	#     writer.write("%s\t%s\n" % (str(preds[i]), str(pred)))
			# 	for i,pred in enumerate(preds_old):
			# 		writer.write( " ".join([str(t) for t in pred]) +"\n")


		#comment ở đây
		if not (args.task_name in ["pride"] and args.do_test):
			with open(output_eval_file, "w") as writer:
				logger.info("***** Eval results {} *****".format(prefix))
				for key in sorted(result.keys()):
					logger.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))

	return result, result_macro


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
	print('args k ', args.k)
	# if args.k <= 0:
	# 	args.k = None
	# if task == "jurgens_binary":
	# 	processor = processors[task](args.k, args.label)
	# else:
	# 	processor = processors[task](args.k)
	if task == "jurgens_binary":
		processor = processors[task](None, args.label)
	else:
		processor = processors[task](None)
	output_mode = output_modes[task]
	# Load data features from cache or dataset file
	if evaluate and test:
		raise ValueError()
	elif evaluate:
		mode = "dev"
	elif test:
		mode = "test"
	else:
		mode = "train"
	cached_features_file = os.path.join(
		args.data_dir,
		"cached_{}_{}_{}_{}".format(
			mode,
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
			str(task),
		),
	)
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", args.data_dir)
		label_list = processor.get_labels()
		if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
			# HACK(label indices are swapped in RoBERTa pretrained model)
			label_list[1], label_list[2] = label_list[2], label_list[1]

		if mode == "train":
			examples = (processor.get_train_examples(args.data_dir))
		elif mode == "dev":
			examples = (processor.get_dev_examples(args.data_dir))
		elif mode == "test":
			examples = (processor.get_test_examples(args.data_dir))

		#examples = (
		#    processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
		#)
		features = convert_examples_to_features(
			examples,
			tokenizer,
			label_list=label_list,
			max_length=args.max_seq_length,
			output_mode=output_mode,
			#pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
			#pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
			#pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
		)
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save(features, cached_features_file)

	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	# Convert to Tensors and build dataset
	# all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	# all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	# if args.model_type != "distilbert":
	#     all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long) if args.model_type \
	#                                                                                                  in ["bert", "xlnet", "albert"] \
	#         else torch.zeros_like(all_input_ids)
	if output_mode == "classification":
		all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
	if output_mode == "multilabel_classification":

		print('ouput model ' , output_mode)
		binary_labels = [] 
		for i,f in enumerate(features):
			label = np.zeros(len(label_list))
			label[f[1]] = 1
			features[i].append(torch.tensor(label, dtype=torch.long))

	elif output_mode == "regression":
		all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
	
	# dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
	# args.k = 0
	# return dataset
	return features


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		required=True,
		help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
	)
	parser.add_argument(
		"--model_type",
		default=None,
		type=str,
		required=True,
		help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help="Path to pre-trained model or shortcut name selected in the list." # + ", ".join(ALL_MODELS),
	)
	parser.add_argument(
		"--task_name",
		default=None,
		type=str,
		required=True,
		help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)

	parser.add_argument(
		"--classification_type",
		default="multiclass",
		type=str,
		required=True,
		help="Whether to run a multilabel or multiclass classification",
	)

	# Other parameters
	parser.add_argument(
		"--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from s3",
	)
	parser.add_argument(
		"--max_seq_length",
		default=128,
		type=int,
		help="The maximum total input sequence length after tokenization. Sequences longer "
		"than this will be truncated, sequences shorter will be padded.",
	)

	parser.add_argument(
		"--n_iter_sent",
		default=1,
		type=int,
	)

	parser.add_argument(
		"--save_model",
		default=0,
		type=int,
	)
	parser.add_argument(
		"--n_iter_word",
		default=1,
		type=int,
	)

	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
	)
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
	)

	parser.add_argument(
		"--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
	)
	parser.add_argument(
		"--k", default=-1, type=int, help="K few shot instances",
	)

	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

	parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument("--alpha", type=float, default=1.0, help="random seed for initialization")


	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
		"See details at https://nvidia.github.io/apex/amp.html",
	)
	parser.add_argument("--label", type=str, default="", help="Target label for binary classification")

	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
	args = parser.parse_args()

	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd

		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
		args.fp16,
	)

	# Set seed
	set_seed(args)

	if args.do_eval and args.do_test:
		raise ValueError()

	args.task_name = args.task_name.lower()
	# print('processors ', processors)
	if args.task_name not in processors:
		raise ValueError("Task not found: %s" % (args.task_name))
	if args.task_name == "jurgens_binary":
		processor = processors[args.task_name](args.k, args.label)
	else:
		processor = processors[args.task_name]()
	print('processor ', processor)
	args.output_mode = output_modes[args.task_name]
	label_list = processor.get_labels()
	num_labels = len(label_list)
	print('num labels >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', num_labels , 'processor ' , processor)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


	if args.task_name not in ["kylel"]:
		config = config_class.from_pretrained(
			args.config_name if args.config_name else args.model_name_or_path,
			num_labels=num_labels,
			finetuning_task=args.task_name,
			cache_dir=args.cache_dir if args.cache_dir else None,
		)
	else:
		config = config_class.from_pretrained(
			args.config_name if args.config_name else args.model_name_or_path,
			finetuning_task=args.task_name,
			num_labels=num_labels,
			cache_dir=args.cache_dir if args.cache_dir else None,
		)

	# tokenizer = tokenizer_class.from_pretrained(
	#     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
	#     do_lower_case=args.do_lower_case,
	#     cache_dir=args.cache_dir if args.cache_dir else None,
	# )
	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# pretrained_model = BertModel.from_pretrained('bert-base-uncased')
	tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
	# pretrained_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
	print('finish load scibert uncased model  ')
	# model = Model(pretrained_model , 768 , 768 , args.n_iter_sent , args.n_iter_word, args.device)
	import pickle 

	with open('checkpoint_best' ,'rb') as f:
		model = pickle.load(f) 
	
	total_params = sum(p.numel() for p in model.parameters())
	print('Total params ' , total_params)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)

	print('Start evaluating ..... ')
	evaluate(args, model, tokenizer, processor = processor , save_output= '.', global_step= 0)
	# Training


if __name__ == "__main__":
	main()
