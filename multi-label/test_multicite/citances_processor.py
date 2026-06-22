import glob
import codecs
import numpy as np
import random
from preprocess import * 
from util_1 import * 
import nltk  as nltk

""" Citances processors and helpers """

import logging
from typing import List, Optional, Union
import json
import jsonlines

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

def citances_convert_examples_to_features(
	examples: List[InputExample],
	tokenizer: PreTrainedTokenizer,
	max_length: Optional[int] = None,
	task=None,
	label_list=None,
	output_mode=None
):
	"""
	Loads a data file into a list of ``InputFeatures``

	Args:
		examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
		tokenizer: Instance of a tokenizer that will tokenize the examples
		max_length: Maximum example length. Defaults to the tokenizer's max_len
		task: GLUE task
		label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
		output_mode: String indicating the output mode. Either ``regression`` or ``classification``

	Returns:
		If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
		containing the task-specific features. If the input is a list of ``InputExamples``, will return
		a list of task-specific ``InputFeatures`` which can be fed to the model.

	"""
	return _convert_examples_to_features(
		examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
	)


def _convert_examples_to_features(
	examples: List[InputExample],
	tokenizer: PreTrainedTokenizer,
	max_length: Optional[int] = None,
	task=None,
	label_list=None,
	output_mode=None,
):
	def tok_split_lm(sents, tokenizer, max_sequence_length = 490):
		tok2sent = [] 
		list_toks = [] 
		# sents = sent_split(context) #list of sents, each
		list_words = []
		for i, sent in enumerate(sents) :
			wds = sent.split() # list of word
			for wd in wds:
				toks = tokenizer.tokenize(wd) #list of toks 
				for tok in toks:
					tok2sent.append(i) #append index of current sentence 
					list_toks.append(tok)
		chunks_toks = [list_toks[x:x+max_sequence_length] for x in range(0, len(list_toks), max_sequence_length)]
		chunks_idx = [tokenizer.convert_tokens_to_ids(t) for t in chunks_toks]
		# chunks_sent_indx  = [tok2sent[x:x+max_sequence_length] for x in range(0, len(tok2sent), max_sequence_length)]
		return tok2sent ,  chunks_idx

	# def clean_noise(string):
	# 	start , end = '<cite>' , '</cite>'
	# 	st_ids = [match.start() for match in re.finditer(start, string)]
	# 	end_ids = [match.start() for match in re.finditer(end, string)]
	# 	if len(st_ids) != len(end_ids) :
	# 		print('error ....')
	# 		return string 
	# 	list_st_remove = [] 
	# 	for i in range(len(st_ids)):
	# 		s , e = st_ids[i] , end_ids[i]
	# 		list_st_remove.append(string[s:(e+len('</cite>'))])
	# 	for remove_str in list_st_remove:
	# 		string = string.replace(remove_str , " ")
	# 	return " ".join(string.split())
	
	def clean_noise(string):
		return string.replace("<cite>", " ").replace("</cite>"," ")

	def sort_list(list_sent):
		num_element = len(list_sent)
		ori_dict = {}
		for sent in list_sent:
			ori_dict[sent['sent_id']] = sent['text']
		template_sent_id = "-".join(list_sent[0]['sent_id'].split('-')[:-1])
		new_list_sent = [] 
		for i in range(num_element):
			new_sent_id = template_sent_id+"-"+str(i+1)
			new_list_sent.append(ori_dict[new_sent_id])
		return new_list_sent
	
	def split_text_with_present(id , text , pre_sent, citation_length):
		if type(text) == list :
			text = text[0]
		if type(text ) != str or type(pre_sent) != list :
			print('error type ', type(text), type(pre_sent))
			print(text)
			return
		a = text 
		orig = norm_string(text)
		text = norm_string(text)

		sorted_sents = [] 	
		for i in range(len(pre_sent)):
			sorted_sents.append((i, pre_sent[i]))
		sorted_sents = sorted(sorted_sents, key=lambda x: len(x[1]) , reverse=True)
		# print('sorted sents ' , sorted_sents)
		result = [] 
		for sent in sorted_sents:
			if norm_string(sent[1]) in text:
				result.append(sent[1])
				text= text.replace(norm_string(sent[1]) , '' , 1)
				# print('current text ', text )
		
		if len(result) > citation_length : 
			out =  result
		else:
			reorder_sents = reorder_se(result , orig)
			out =  reorder_sents

		return out

	if max_length is None:
		max_length = tokenizer.max_len

	if task is not None:
		processor = citances_processors[task]()
		if label_list is None:
			label_list = processor.get_labels()
			logger.info("Using label list %s for task %s" % (label_list, task))
		if output_mode is None:
			output_mode = citances_output_modes[task]
			logger.info("Using output mode %s for task %s" % (output_mode, task))

	if label_list is not None:
		label_map = {label: i for i, label in enumerate(label_list)}

	def label_from_example(example: InputExample) -> Union[int, float]:
		if output_mode == "classification":
			return label_map[example.label]
		if output_mode == "multilabel_classification":
			return [label_map[l] for l in example.label]
		raise KeyError(output_mode)

	if label_list is not None:
		labels = [label_from_example(example) for example in examples]
		#labels = [example.label for example in examples]

	# with open('data_new_1.json' , 'r', encoding='UTF-8') as f_full:
	# 	full_data =  json.load(f_full)

	list_chunk2ids  = [] # we have 4 list to save 
	with open('check.json' ,'w') as f_w:

		check_datas = [] 
		for i, example in enumerate(examples):

			guid = example.guid
			text = example.text_a

			tokens = tokenizer.tokenize(text) # list token 

			indexes= tokenizer.convert_tokens_to_ids(tokens) #list index 

			list_chunk2ids.append(indexes)
			
			check_data = {'id':i , 'text':text , 'num_tok':len(tokens)}

			check_datas.append(check_data)
		
		json.dump(check_datas , f_w, indent= 2)

	features = []
	for i in range(len(examples)):
		# inputs = {k: batch_encoding[k][i] for k in batch_encoding}
		if label_list is not None:
			features.append([list_chunk2ids[i], labels[i] ])
		# else:
		# 	features.append([list_tok2sent[i], list_chunk2ids[i]])

	return features


class OurClassificationProcessor(DataProcessor):
	"""Processor for the classification version of our data set."""
	def __init__(self, k=None):
		super()
		self.k = k

	def _read_json(self, path):
		with open(path) as f_in:
			return json.load(f_in)

	def get_train_examples(self, path=""):
		"""See base class."""
		path += "/train.json"
		logger.info("LOOKING AT {}".format(path))
		if self.k is None:
			return self._create_examples(self._read_json(path), "train")
		else:
			return self._create_examples(self._read_json(path), "train")[:self.k]

	def get_dev_examples(self, path=""):
		"""See base class."""
		path += "/dev.json"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_json(path), "dev")

	def get_test_examples(self, path=""):
		"""See base class."""
		logger.info("LOOKING AT {}".format(path))
		path += "/test.json"
		return self._create_examples(self._read_json(path), "test")

	def get_labels(self):
		"""See base class."""
		return ["motivation", "background", "uses",  "extends", "similarities",
				"differences",  "future_work"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for i, line in enumerate(lines):
			guid = line["id"]#"%s-%s" % (set_type, i)
			text = " ".join(line["x"]) if isinstance(line["x"], list) else line["x"]
			label = line["y"].split(" ")
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

		return examples


class OurClassificationProcessorJurgens(DataProcessor):
	"""Processor for the classification version of our data set."""
	def __init__(self, k=None):
		super()
		self.k = k

	def _read_json(self, path):
		with open(path) as f_in:
			return json.load(f_in)

	def get_train_examples(self, path=""):
		"""See base class."""
		path += "/train.json"
		logger.info("LOOKING AT {}".format(path))
		if self.k is None:
			return self._create_examples(self._read_json(path), "train")
		else:
			return self._create_examples(self._read_json(path), "train")[:self.k]

	def get_dev_examples(self, path=""):
		"""See base class."""
		path += "/dev.json"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_json(path), "dev")

	def get_test_examples(self, path=""):
		"""See base class."""
		logger.info("LOOKING AT {}".format(path))
		path += "/test.json"
		return self._create_examples(self._read_json(path), "test")

	def get_labels(self):
		"""See base class."""
		return ["Uses", "Motivation", "Future", "Extends", "CompareOrContrast", "Background"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		labels = {
			"motivation": "Motivation",
			"background": "Background",
			"uses": "Uses",
			"extends": "Extends",
			"similarities": "CompareOrContrast",
			"differences": "CompareOrContrast",
			"future_work": "Future"
		}
		for i, line in enumerate(lines):
			guid = line["id"]#"%s-%s" % (set_type, i)
			text = " ".join(line["x"]) if isinstance(line["x"], list) else line["x"]
			label = [labels[l] for l in line["y"].split(" ")] # we take all labels
			#label = labels[line["y"].split(" ")[0]] # we only take the first label here, not for training!
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
		return examples


class JurgensProcessor(DataProcessor):
	"""Processor for the cohan data set."""
	def __init__(self, k=None):
		super()
		self.k = k

	def _read_jsonl(self, path):
		with jsonlines.open(path) as reader:
			return list(reader)

	def get_train_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "train.jsonl"
		logger.info("LOOKING AT {}".format(path))
		if self.k is None:
			return self._create_examples(self._read_jsonl(path), "train")
		else:
			return self._create_examples(self._read_jsonl(path), "train")[:self.k]

	def get_dev_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "dev.jsonl"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_jsonl(path), "dev")

	def get_test_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "test.jsonl"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_jsonl(path), "test")

	def get_labels(self):
		"""See base class."""
		return ["Uses", "Motivation", "Future", "Extends","CompareOrContrast", "Background"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for i, line in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text = line["text"]
			label = line["intent"]
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

		return examples


class JurgensBinaryProcessor(DataProcessor):
	"""Processor for the cohan data set."""
	def __init__(self, k=None, label=None):
		super()
		self.k = k
		self.label = label

	def _read_jsonl(self, path):
		with jsonlines.open(path) as reader:
			return list(reader)

	def get_train_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "train.jsonl"
		logger.info("LOOKING AT {}".format(path))
		if self.k is None:
			return self._create_examples(self._read_jsonl(path), "train")
		else:
			return self._create_examples(self._read_jsonl(path), "train")[:self.k]

	def get_dev_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "dev.jsonl"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_jsonl(path), "dev")

	def get_test_examples(self, path="./data/jurgens/"):
		"""See base class."""
		path += "test.jsonl"
		logger.info("LOOKING AT {}".format(path))
		return self._create_examples(self._read_jsonl(path), "test")

	def get_labels(self):
		"""See base class."""
		return ["True", "False"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for i, line in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text = line["text"]
			if line["intent"] == self.label:
				label = "True"
			else:
				label = "False"
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

		return examples




tasks_num_labels = {
	"jurgens": 6,
	"ours": 7,
	"ours_jurgens": 6,
	"jurgens_binary": 2
}

citances_processors = {
	"ours": OurClassificationProcessor,
	"jurgens": JurgensProcessor,
	"jurgens_binary": JurgensBinaryProcessor,
	"ours_jurgens": OurClassificationProcessorJurgens
}

citances_output_modes = {
	"ours": "multilabel_classification",
	"ours_jurgens": "classification",
	"jurgens_binary": "classification",
	"jurgens": "classification",
}

