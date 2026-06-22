# from nltk.tokenize import word_tokenize
# from handle_text_bert import sent_tokenize
# from nltk.tokenize import sent_tokenize
import json
import os
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from tqdm import tqdm
FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
				'-', '--', '|', '\/']
FILTERWORD.extend(punctuations)
from model import *
import torch 
import json 

def load_data_json(file_json , list_label = ['uses', 'background', 'similarities', 'extends', 'motivation', 'differences', 'future_work']):
	with open(file_json , 'r', encoding='UTF-8') as f_full:
		datas =  json.load(f_full) #data = list of json object 
	X = [] 
	Y = []
	for data in datas: 
		X.append(data['x']) #data['x'] is list of sents 
		Y.append(data['y'])
	#create multi class label for training
	labels = []
	for y in Y: 
		init_label = [0] * len(list_label) 
		func = [t.strip() for t in y.split()]
		for i in range(len(list_label)):
			if list_label[i] in func:
				init_label[i] = 1
		labels.append(init_label)
	return X , labels 
		
def preprocess(text, max_seq_length, is_training, tokenizer):
	#process text before feeding to the Word Encoder. (BERT , SciBERT)
	tok_to_orig_index = []
	orig_to_tok_index = []
	all_doc_tokens = []
	word_to_sent_index = []
	
	sents = sent_tokenize(text)  #split citation context into list of sents
	if sum([len(sent.split()) for sent in sents]) != len(text.split()):
		print('erorr tokenize sent ')
		return 
	current_num_word = 0 
	for i , sent in enumerate(
	):
		words = sent.split()
		for word in words :
			word_to_sent_index.append(i)
			
	subword2sent = []
	for (i, token) in enumerate(text.split()):
		orig_to_tok_index.append(len(all_doc_tokens))
		if tokenizer.__class__.__name__ in [
			"RobertaTokenizer",
		]:
			sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
			for sub_token in sub_tokens:
				subword2sent.append(word_to_sent_index[i])
		else:
			#for BERT 
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				subword2sent.append(word_to_sent_index[i])


		for sub_token in sub_tokens:
			tok_to_orig_index.append(i)
			all_doc_tokens.append(sub_token)
		
	sent2subword = {}
	for i in range(len(subword2sent)):
		if subword2sent[i] not in sent2subword:
			sent2subword[subword2sent[i]] = [i]
		else:
			sent2subword[subword2sent[i]].append(i)
			
	spans = []
	truncated_query = tokenizer.encode(
		text , add_special_tokens=False, truncation=True, max_length=max_seq_length
	)
	
	tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
	sequence_added_tokens = (
		tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
		if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
		else tokenizer.model_max_length - tokenizer.max_len_single_sentence
	)
	
	sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
	span_doc_tokens = all_doc_tokens
	len_doc = len(all_doc_tokens)
	#split doc to multiple chunk with max len context 
	contexts = [ all_doc_tokens[i:i+max_seq_length] for i in range(0 , len_doc , max_seq_length)]
	subword2sents = [subword2sent[i:i+max_seq_length] for i in range(0 , len_doc, max_seq_length)]
	
	# for i in range(len(contexts)):
	# 	context = contexts[i] #context of this chunk. We split document to multiple chunks due to limitation of BERT
	# 	sub2sent = subword2sents[i] #subword 2 sent only in one chunk 
	# 	if tokenizer.padding_side == "right": 
	# 		texts = truncated_query
	# 		pairs = context 
	# 		truncation = TruncationStrategy.ONLY_SECOND.value
	# 		encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
	# 		texts,
	# 		pairs,
	# 		truncation=truncation,
	# 		padding=padding_strategy,
	# 		max_length=max_seq_length,
	# 		return_token_type_ids=True,
	# 	)
		# if tokenizer.pad_token_id in encoded_dict["input_ids"]:
		# 	if tokenizer.padding_side == "right":
		# 		non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
		# else:
		# 	non_padded_ids = encoded_dict["input_ids"]

		# tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
		# encoded_dict["paragraph_len"] = len(context)
		# encoded_dict["tokens"] = tokens
		# encoded_dict['sub2sent'] = sub2sent
		# spans.append(encoded_dict)
		
	return 


def sent_split(context):
	#split context to list of sent , each sent is further splitted into list of subword 
	sents = sent_tokenize(context)
	return sents

def tok_split_lm(sents, tokenizer, max_sequence_length = 500):
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
	print('list toks' , list_toks)
	chunks_toks = [list_toks[x:x+max_sequence_length] for x in range(0, len(list_toks), max_sequence_length)]
	chunks_idx = [tokenizer.convert_tokens_to_ids(t) for t in chunks_toks]
	# chunks_sent_indx  = [tok2sent[x:x+max_sequence_length] for x in range(0, len(tok2sent), max_sequence_length)]
	return tok2sent ,  chunks_idx

def load_bert_tokenizer():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return tokenizer

def load_bert_model():
	model = BertModel.from_pretrained('bert-base-uncased')
	return model

def load_roberta_tokenizer():
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	return tokenizer

def load_roberta_model():
	model = RobertaModel.from_pretrained('roberta-base')
	return model

if __name__ == "__main__":

	# X , y = load_data_json(args)
	# sample_X = X[0] #list of sentence 
	# sample_y = y[0] 

# 	context = """ 
# To study impact of stacking multiple feature learning modules trained using different self-supervised loss functions, we test the discrete and continuous BERT pre-training approaches on spectral features and on learned acoustic representations, showing synergitic behaviour between acoustically motivated and masked language model loss functions. In low-resource conditions using only 10 hours of labeled data, we achieve Word Error Rates (WER) of 10.2% and 23.5% on the standard test \"clean\" and \"other\" benchmarks of the Librispeech dataset, which is almost on bar with previously published work that uses 10 times more labeled data. Moreover, compared to previous work that uses two models in tandem <cite>(Baevski et al., 2019b)</cite> , by using one model for both BERT pre-trainining and fine-tuning, our model provides an average relative WER reduction of 9%. 1 ---------------------------------- **INTRODUCTION** Representation learning has been an active research area for more than 30 years (Hinton et al., 1986) , with the goal of learning high level representations which separates different explanatory factors of the phenomena represented by the input data (LeCun et al., 2015; Bengio et al., 2013) . Disentangled representations provide models with exponentially higher ability to generalize, using little amount of labels, to new conditions by combining multiple sources of variations. 1 We will open source the code for our models.
# """
	# sents = sent_split(context)
	# print(len(sents))
	# for i in range(len(sents)):
	# 	print(sents[i])
	# 	print('--------')
	
	# print(len(context.split()))
	sents = ['uses', 'background', 'similarities', 'extends', 'motivation', 'differences', 'future_work']
	tokenizer = load_roberta_tokenizer()
	lm = load_roberta_model()
	print('sample X ' , sents)
	tok2sent ,  chunks_idx = tok_split_lm(sents , tokenizer, max_sequence_length=500)
	print('tok2 sent ' , tok2sent)
	print('chunks id ', chunks_idx)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model(pretrained_model= lm , word_emb_dim=768 , sent_emb_dim= 768 , device=device)
	model = model.to(device)
	doc_logit = model(chunks_idx , tok2sent)
	output = loss(torch.tensor(doc_logit), torch.tensor(sample_y).float())
	print(output)

