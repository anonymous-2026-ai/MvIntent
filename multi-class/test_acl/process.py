import json
from handle_text_bert import sent_tokenize

def sent_split(text):
	list_line = text.split('\n')
	sents = [] 
	for sent in list_line:
		chunks = sent_tokenize(sent)
		for chunk in chunks:
			sents.append(chunk)
	#check 
	if len(text.split()) != sum([len(sent.split()) for sent in sents]) :
		print('error when split sentence .........')
		print(len(text.split()) , sum([len(sent.split()) for sent in sents]))
	return sents 

# with open('./train.jsonl', 'r') as json_file:
#     json_list = list(json_file)

# for i, json_str in enumerate(json_list):
#     json_object = json.loads(json_str)
#     text = json_object['string']
#     sents = sent_split(text)
#     # if len(sents) > 1 :
#     #     print('--------')
#     #     print(i , sents , len(sents))
#     #     print(len(text.split()) , sum([len(sent.split()) for sent in sents]))
#     print(json_object['label'])
def convert_jsonl2json(type_file):
	with open(type_file+'.jsonl', 'r') as jsonl_file:
		json_list = list(jsonl_file)
		out = [] 
		id = 0 
		with open(type_file+'.json' , 'w') as out_file :
			for i, json_str in enumerate(json_list):
				json_object = json.loads(json_str)
				# text = json_object['string']
				# label = json_object['label']
				doc  = {'id': id , 'x': json_object['string'] , 'y':json_object['label']}
				id +=  1 
				out.append(doc)
			json.dump(out, out_file, indent=4)

#convert_jsonl2json('test')
#convert_jsonl2json('dev')
