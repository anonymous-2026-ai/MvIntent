import spacy
from collections import deque
from object_extract import extract_object
from subject_extract import extract_subject
from transformers import AutoTokenizer


def recover_noun_chunk_from_index(list_noun_chunk_index, index):
	if index == None :
		return None 
	for noun_chunk_index in list_noun_chunk_index:
		if index in noun_chunk_index:
			return noun_chunk_index
	return [index]

def map_index_to_text(tokens, list_index):
	if list_index == None :
		return ""
	else:
		return " ".join([tokens[t].text for t in list_index])

def extract_triple_full_index(nlp, text):
	dict_object , dict_subject = extract_object(nlp, text) , extract_subject(nlp, text)
	# tokens = [token for token in nlp(text )]
	triples = [] 
	for verb_index in dict_subject.keys():
		triples.append((dict_subject[verb_index] , verb_index, dict_object[verb_index]))
	list_noun_chunk_index = [ ]
	for chunk in nlp(text).noun_chunks:
		list_noun_chunk_index.append([token.i for token in chunk])
	list_triple_full_index = [] 
	for triple in triples:
		s_recover = recover_noun_chunk_from_index(list_noun_chunk_index , triple[0])
		p_recover = recover_noun_chunk_from_index(list_noun_chunk_index , triple[1])
		o_recover = recover_noun_chunk_from_index(list_noun_chunk_index , triple[2])
		list_triple_full_index.append((s_recover , p_recover, o_recover))
	
	return list_triple_full_index

def find_indexes(noun_chunk_token , context_token, index_return):

	indices = []

	for i in range(len(context_token) - len(noun_chunk_token) + 1):
		if context_token[i:i+len(noun_chunk_token)] == noun_chunk_token:
			indices.append(list(range(i, i+len(noun_chunk_token))))
	
	if len(indices) > 0:
		return indices[index_return]
	else:
		return []

def handle_triple(triple, tokenizer):
	s , p , o = triple[0] , triple[1] , triple[2]
	triple_tokens = [] 
	for t in [s , p , o]:
		# print('t', t)
		if t != None :
			triple_tokens.append(tokenizer.tokenize(t))
		else:
			triple_tokens.append(None)
	return triple_tokens


def get_occurrence_from_index(tokens, chunk_index):
	"""
	Xác định chunk (list index) là lần xuất hiện thứ mấy trong toàn text.
	- tokens: list token của toàn bài
	- chunk_index: list index của chunk cần tìm, ví dụ [3,4]
	"""

	# 1. Lấy nội dung chunk
	
	if chunk_index != None :
		chunk_tokens = tokens[chunk_index[0] : chunk_index[-1] + 1]

		# 2. Tìm tất cả vị trí mà chuỗi này xuất hiện trong tokens
		occurrences = []
		L = len(chunk_tokens)

		for start in range(len(tokens) - L + 1):
			if tokens[start:start+L] == chunk_tokens:
				occurrences.append(list(range(start, start+L)))

		# 3. Chunk hiện tại là occurrence thứ mấy?
		for i, occ in enumerate(occurrences):
			if occ == chunk_index:
				return i
	else:
		return None

def find_indexes(chunk_token , context_token, index_return):

	if index_return == None :
		return []
	indices = []

	for i in range(len(context_token) - len(chunk_token) + 1):
		if context_token[i:i+len(chunk_token)] == chunk_token:
			indices.append(list(range(i, i+len(chunk_token))))
	
	if len(indices) > 0:
		return indices[index_return]
	else:
		return []

def map_chunk_index_to_text(spacy_tokens, chunk_index):
	if chunk_index != None :
		return " ".join([spacy_tokens[t].text for t in chunk_index])
	else:
		return None 



def create_triple_to_index_text(text, spacy_nlp, lm_tokenizer):
	tokens = [token for token in spacy_nlp(text)]
	list_triple_full_index = extract_triple_full_index(spacy_nlp ,text)

	lm_tokens = lm_tokenizer.tokenize(text)
	out = {'triple_text':[] , 'triple_index':[] }
	for triple_full_index in list_triple_full_index:
		triple_full_text = [map_chunk_index_to_text(tokens , t) for t in triple_full_index]
		triple_full_current_index = [get_occurrence_from_index(tokens , t) for t in triple_full_index]
		
		# print('triple full index ',triple_full_index)
		# print('triple full text ' , triple_full_text)
		# print('triple full current index ', triple_full_current_index ) 
		triple_full_token = [lm_tokenizer.tokenize(t) if t != None else [] for t in triple_full_text ]

		# print('triple full tokens ' , triple_full_token)
		triple_full_token_lm_index = [find_indexes(triple_full_token[i] , lm_tokens , index_return=triple_full_current_index[i]) for i in range(len(triple_full_token))]
		# print('triple_full_token_lm_index' , triple_full_token_lm_index)
		# print('checked lm tokens' , [" ".join([lm_tokens[i] for i in t]) for t in triple_full_token_lm_index])
		out['triple_index'].append(triple_full_token_lm_index)
		out['triple_text'].append(triple_full_text)
	return out 

def create_triple_to_index_file(filein, fileout , spacy_nlp, lm_tokenizer):
	import json 
	out = [] 
	with open(filein ,'r') as f_r, open(fileout , 'w') as f_w:
		full_data = json.load(f_r)
		for i, doc in enumerate(full_data) : 
			print( i , ' / ', len(full_data))
			id , x , y  = doc["id"] , doc['x'] , doc['y']
			if type(x) != str:
				x = x[0]
			x = x.replace('</cite>','').replace('<cite>','')
			triple_info = create_triple_to_index_text(x , spacy_nlp , lm_tokenizer)
			out.append({'id':id , 'x':x , 'triple_info':triple_info})
		json.dump(out, f_w, indent = 2)

lm_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

spacy_nlp = spacy.load("en_core_web_sm")
with open('ner_example.txt' ,'r') as f_r:
	text = f_r.read()

# for dataset in ['train' ,'dev', 'test']:
# 	filein , fileout  = "../data_acl_arc/"+dataset+".json","../data_acl_arc/"+dataset+"_triple.json"
# 	create_triple_to_index_file(filein , fileout , spacy_nlp , lm_tokenizer)
# out = create_triple_to_index_text(text, spacy_nlp, lm_tokenizer)
print('text ' , text)
# for t in out['triple_text']:
# 	print(t)
doc = spacy_nlp(text)

# in ra tất cả noun chunks
for chunk in doc.noun_chunks:
    print(chunk.text)