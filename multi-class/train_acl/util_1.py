import json
import re
import itertools
from itertools import permutations

def norm_string(string):
	return string.replace('</cite>','').replace('<cite>','').strip()

def reorder_se(sents , orig):
	num_elements = len(sents)
	list_per = list(permutations(range(num_elements)))
	for i, set_index in enumerate(list_per):
		# print(i)
		list_sent = [sents[i] for i in set_index]
		if norm_string(" ".join(list_sent)) in orig:
			return list_sent
	return sents

def split_text_baseon_list_sentence(id, text , pre_sents, citation_length):
	orig = text
	isValid = False 
	#input: text , sents are pre_defined as the authors and the dataset 
	#ouput return list of sents which constitute the text
	if type(text) == list :
		orig = text[0]
	if type(text ) != str or type(pre_sents) != list :
		print('error type ', type(text), type(pre_sents))
		print(text)
		return   
	result = [] 
	curr_text = [] 
	for i , sent in enumerate(pre_sents):
		if  norm_string(orig) in norm_string(" ".join(pre_sents[i:i+citation_length])):
			isValid = True 
			result =  pre_sents[i:i+citation_length]
		# else:
		# 	print('test ' , " ".join(pre_sents[i:i+citation_length]))
		# 	print('orig' , orig)
	if isValid == False:
		print('not exist ')
		print(id)
		print(orig)
	return result

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

def a():
	with open('data_new.json', 'r' , encoding='UTF-8') as f_in:
		datas = json.load(f_in)
	result = {}
	keys = list(datas.keys())
	for key in keys :
		data = datas[key]
		#change the name of key and keep the value of data 
		new_key = key.split('_')[1]
		result[new_key] = data 
	with open('data_new_1.json', 'w', encoding='UTF-8') as f:
		json.dump(result, f , indent=4)

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
	# if norm_string(" ".join(out)).strip() not in  orig.strip():
	# 	print('---------')
	# 	print('len result' , len(result))
	# 	print(norm_string(" ".join(out)))
	# 	print('\n')
	# 	print(orig)
	# 	print('\n')
	# 	print('id ', id)
		# if len(result) <= citation_length:
		# 	print('\n')
		# 	print('reorder ' , reorder_se(result , orig))
		# print('\n')
		# print(result)
	return out

def process_text2sent_file(input_json , full_data_json, citation_length ):
	count_error = 0 
	total = 0 
	#input json: classification_3_context , 
	#full_data_json: data_new.json # full data when replace <span > by <cite> 
	#output_json: classification_3_context_new
	#load full data json 
	with open(full_data_json , 'r', encoding='UTF-8') as f_full:
		full_data =  json.load(f_full)

	f_in = open(input_json)
	# output_json = input_json + '_new'
	datas = json.load(f_in) #list of json file 
	#each json = 'x' , 'y' , 'id'
	result = [] 
	l = len(datas)
	for i, data in enumerate(datas):
		total += 1
		id = data['id']
		x = data['x'] # citation context, which need to be splitted base on list of sentence in original document 
		y = data['y']
		#base on id of sentence, find the relevant doc 
		doctid = id.split('_')[0]
		if doctid not in full_data:
			count_error += 1
			print('id ', id, 'doc id ', doctid)
			continue
		#get list of relevant sentid in 
		# print('id', id , 'doctid ', doctid)
		full_doc = full_data[doctid]['x'] # list of json, json = { sentid , text  }
		full_doc_sorted = sort_list(full_doc)
		#split 
		out = split_text_with_present(id , x , full_doc_sorted,citation_length) # split citation context into list of sentences 
		#keep original structure of json file 
		# print('out ', out)
		result.append({ 'id':id , 'x':out, 'y':y  })

	print('number error ', count_error , total )
	output_json = input_json.split('.')[0]+'_proposed.json'
	print(output_json)
	with open(output_json, 'w', encoding='UTF-8') as f:
		json.dump(result, f , indent=4)

# input_json = '.\multicite\data\classification_10_context\dev.json'
# full_data_json = 'data_new.json'
# process_text2sent_file('multicite/data/classification_gold_context/train.json' , 'data_new_1.json', 10 )