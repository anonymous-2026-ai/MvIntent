from transformers import BertModel
import torch.nn as nn
import torch 
from Doc_representation import * 
import numpy as np
from TripleEnhanceLayer import * 
# from TripleEnrichLayer import * 
# from TripleAttention import *
# from TripleIntentInteraction import * 
# from TripleIntentWeight import * 
from TripleWeight import * 
from torch.nn.utils.rnn import pad_sequence


class Model(nn.Module):

	def __init__(self, pretrained_model, word_emb_dim, n_iter_sent , n_iter_word, device):
		super().__init__()
		self.device = device
		self.pretrained_model = pretrained_model
		self.word_emb_dim = word_emb_dim 
		self.linear_pool = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.doc_pool = Doc_representation(self.word_emb_dim, n_func = 7 , device = self.device)
		self.triple_enhance = TripleEnhanceLayer(self.word_emb_dim  , n_iter_word , self.device)
		# self.triple_enrich = TripleEnrichLayer(self.word_emb_dim, self.device)
		# self.triple_intent_interact = TripleIntentInteraction(self.word_emb_dim , n_iter = 1)
		self.triple_weight = TripleWeight(self.word_emb_dim)
		self.triple_word_integrate = nn.Linear(2* 768 , 1)
		self.triple_linear = nn.Linear(768, 768)
		self.intent_linears = nn.Linear(768 ,7 , bias=False)
		self.triple_intent_linear = nn.Linear(768, 768)


	def pool_sent_embed(self , last_hidden_state , features,  batch_index):
		batch_feature = [features[t] for t in batch_index]
		ques_embed , list_sent_emb , batch_bound_sents = self.pool_embed(last_hidden_state , batch_feature) #init embed contain question embedding and list of sentence embedding 
		return ques_embed , list_sent_emb , batch_bound_sents

	def forward_pretrained_with_multilen_seq(self, list_seq , return_type = 'cls'):
		pad_id = 0
		# 4) Convert sang tensor và pad để batch
		seqs = [torch.tensor(ids, dtype=torch.long) for ids in list_seq]
		input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, Lmax)

		# 5) attention_mask: 1 cho token thật, 0 cho pad
		attention_mask = (input_ids != pad_id).long()  # (B, Lmax)

		# 6) token_type_ids (segment ids) - SciBERT vẫn nhận như BERT
		#    Nếu bạn không dùng sentence pair (A/B) thì để toàn 0 là chuẩn
		token_type_ids = torch.zeros_like(input_ids)

		# 7) Đẩy lên device và forward 1 lần cho cả batch
		batch = {
			"input_ids": input_ids.to(self.device),
			"attention_mask": attention_mask.to(self.device),
			"token_type_ids": token_type_ids.to(self.device),
		}

		outputs = self.pretrained_model(**batch)
		last_hidden_state = outputs.last_hidden_state      # (B, Lmax, 768)
		if return_type == 'cls':
			out  = last_hidden_state[ :, 0, :] #(B,  768)
			return self.triple_linear(out)
		elif return_type =='tokens':
			out = last_hidden_state[: , 1:-1 , :] # (B , seq len , 768 )
			return out 

	# def forward(self, chunks_idx , tok2sent_ind ,  labels_idx , tok2label, tokenizer , print_att = False):
	def forward(self, list_chunks_idx , list_triple_indexes , data_graph,  tokenizer):

		#create batch
		list_batch_index, list_len_chunk = [] , [] 
		for i in range(len(list_chunks_idx)):
			chunk_idx , triple_indexes = list_chunks_idx[i] , list_triple_indexes[i]
			if triple_indexes == []:
				list_batch_index.append([101] + chunk_idx  +  [102])
				list_len_chunk.append(1)
			else:
				list_len_chunk.append(len(triple_indexes))
				for j in range(len(triple_indexes)):
					list_batch_index.append([101] +  tokenizer.encode(triple_indexes[j] , add_special_tokens=False )  +  [102] )
		batch_triple_embed = self.forward_pretrained_with_multilen_seq(list_batch_index, return_type='cls') # shape (Num triple in whole batch , 768)

		list_triple_embed = [] 
		start = 0
		for s in list_len_chunk:
			list_triple_embed.append(batch_triple_embed[start:start + s])
			start += s

		#compute word-level information 
		word_level_index =  [] 
		for i in range(len(list_chunks_idx)):
			chunk_idx= list_chunks_idx[i]
			word_level_index.append([101] + chunk_idx  +  [102])
		word_level_embed = self.forward_pretrained_with_multilen_seq(word_level_index , return_type = 'tokens') #shape (batch size , 768)

		list_tok_embed = [] 
		for i in range(word_level_embed.shape[0]):
			list_tok_embed.append(word_level_embed[i , : len(list_chunks_idx[i]) , :])

		# list_triple_intent_embeds, list_triple_sigmoid = [] , []
		# for i in range(len(list_triple_embed)):
		# 	# intent_weight, triple_weight , triple_intent_att  = self.triple_intent_weight(list_triple_embed[i] , word_level_embed[i])
		# 	triple_embeddings , triple_sigmoids  = self.triple_intent_weight(list_triple_embed[i], self.intent_linears)
		# 	list_triple_intent_embeds.append(triple_embeddings)
		# 	list_triple_sigmoid.append(triple_sigmoids)
		
		intent_embeds = self.intent_linears.weight
		list_enhanced_triple , list_intent_weight = [] , []
		for i in range(len(list_triple_embed)):
			triple_hids, intent_weight = self.triple_enhance(list_triple_embed[i] , list_tok_embed[i], intent_embeds , data_graph[i])
			list_enhanced_triple.append(triple_hids)
			list_intent_weight.append(intent_weight)

		list_overall_triple , list_triple_weight = [] , []  
		for i in range(len(list_enhanced_triple)):
			overall_triple , triple_weight = self.triple_weight(list_enhanced_triple[i])
			list_overall_triple.append(overall_triple)
			list_triple_weight.append(triple_weight)
		list_overall_triple = torch.stack(list_overall_triple , dim = 0 ) #shape (B , 768 )

		# intent_weight = self.intent_linears(list_overall_triple) # (bs , 7 )
		intent_weight = self.intent_linears(list_overall_triple) # (bs , 7 )
		# list_intent_weight  = torch.stack(list_intent_weight , dim= 0 )
		return intent_weight, list_triple_weight  , list_intent_weight
	
	def list_triple_tok_pool(self, toks_embed , list_triple_indexes):

		def pool_single_indexs(toks_embed , list_index): # [] 
			element_tensor = toks_embed[list_index]
			return element_tensor #shape ( len(list_index) , 768)
		
		def pool_list_indexes(toks_embed , list_list_index): # [ [] , [] ]
			out = [] 
			for list_index in list_list_index:
				element_embed = pool_single_indexs(toks_embed , list_index) # (len_list_index, 768) 
				out.append(element_embed)
			out =  torch.stack( out , dim = 0 ) #shape (Num_appear , len list index , 768)
			return torch.mean(out , dim = 0 ) #shape (len list index, 768)
		
		def triple_tok_pool(toks_embed , triple_index): # [ [ [] , []  ] , [ [] ] , [ [] , [] , []  , [] ] ]
			triple_cat = [] 
			for i in range(len(triple_index)):
				element_index = triple_index[i]  #element index: list of list index 
				if element_index != [] :
					triple_cat.append(pool_list_indexes(toks_embed , element_index)) #shape (len list index , 768)

			if len(triple_cat) > 0 :
				triple_cat = torch.cat(triple_cat , dim = 0 ) #shape (len triple = len subject + len predicate + len object , 768)
				return triple_cat 
			else:
				print('Null triple ', triple_index)
				return None 

		list_triple_tok_pool = [] 
		for triple_index in list_triple_indexes:
			triple_tok_embed = triple_tok_pool(toks_embed , triple_index) #shape (len triple , 768 )
			if  triple_tok_embed != None :
				list_triple_tok_pool.append(triple_tok_embed) # list of (len triple , 768 )
		return list_triple_tok_pool # list of [len triple , 768]
	
	def get_label_hidden(self, labels_idx):
		label_idx = labels_idx[0]
		result = [] 
		for idx in label_idx:
			result.append(torch.tensor(self.pretrained_model.embeddings.word_embeddings(torch.tensor(idx).to(self.device))))
		return torch.stack(result, dim = 0)

	def extract_tok_label_emb(self,tok_hidden_state):
		#tok hidden state is list of tensor , each tensor shape (num label , num token , hidden dim  )
		list_tok_emb = [] 
		list_CLS_emb = [] 
		for t in tok_hidden_state: 
			#t shape (num label , num token+  1 +2  , hidden dim )
			tok_emb =  t[: , 1:-1,:] #shape ( num label, num tok in context , word dim )
			# func_emb = t[ : , -2, :].squeeze(1) # shape (num label, word dim)
			cls_emb = t[:,0:1,:]

			tok_emb = torch.mean(tok_emb , dim = 0 ) # shape (num tok in context , word dim)
			cls_emb= torch.mean(cls_emb , dim = 0 )

			list_tok_emb.append(tok_emb)
			list_CLS_emb.append(cls_emb)

			# print('cls shape ' , cls_emb.shape , 'tok shape ' , tok_emb.shape)

		all_tok_emb = torch.cat(list_tok_emb , dim = 0) #shape (num tok in doc, word dim )
		all_cls_emb = torch.cat(list_CLS_emb , dim = 0 ) #shape (num chunk , word dim)

		# print('all cls shape ' , all_cls_emb.shape)
		return all_cls_emb , all_tok_emb 

	def get_label_hidden(self, labels_idx):
		label_idx = labels_idx[0]
		result = [] 
		for idx in label_idx:
			result.append(torch.tensor(self.pretrained_model.embeddings.word_embeddings(torch.tensor(idx).to(self.device))))
		return torch.stack(result, dim = 0)	

	def forward_attention(self, input ):
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(len(input)):
			# input_id= all_input_ids[i, : ].unsqueeze(0)
			x  = torch.tensor(input[i] , dtype=torch.long).to(self.device).unsqueeze(0) #shape (1 , N_word)
			t = self.pretrained_model(x, output_attentions=True, return_dict=True)
			attention  = t.attentions[-1][0][0][-10:-1, :]
			output.append(attention)
		return output 
	
	def forward_pretrained_tok(self, input ):
		#input is list of list of list (num chunk , num label , num token )
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)

		x  = torch.tensor(input , dtype=torch.long).to(self.device) #shape (num label , N_word)
		t = self.pretrained_model(x.unsqueeze(0))[0] #shape (1 , N_word , word dim)

		return t.squeeze(0)  #shape (N_word, word_dim )

	# def forward_pretrained_tok(self, input ):
	# 	#input is list of list of list (num chunk , num label , num token )
	# 	#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
	# 	# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
	# 	output = []
	# 	for i in range(len(input)):
	# 		#for chunk 
	# 		# input_id= all_input_ids[i, : ].unsqueeze(0)
	# 		# x  = torch.tensor(input[i] , dtype=torch.long).to(self.device) #shape (num label , N_word)
	# 		# print('x shape ', x.shape )
	# 		chunk_hid = []
	# 		for j in range(len(input[i])):
	# 			index = input[i][j]
	# 			# input_id= index.unsqueeze(0)
	# 			# print(index)
	# 			x  = torch.tensor(index , dtype=torch.long).to(self.device).unsqueeze(0)
	# 			t = self.pretrained_model(x)[0] #shape (1 , N_word , word dim)
	# 			# print('t shape ', t.shape)
	# 			chunk_hid.append(t.squeeze(0)) #(N word , word dim)
	# 		chunk_hid = torch.stack(chunk_hid) # N label , N word , word dim
	# 		output.append(chunk_hid)
	# 	return output 
	
	def forward_pretrained_label(self, input ):
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		x  = torch.tensor(input , dtype=torch.long).to(self.device).unsqueeze(0) #shape (1 , N_word)
		t = self.pretrained_model(x)[0] #shape (1 , N_word , word dim)
		return t.squeeze(0)

	def sent_pool(self, tok_embed, tok2sent_ind ,  method_pool = 'mean'):
		#tok_embed: shape (sequen_length , hidden size)
		bound_sents = find_boudary_sents(tok2sent_ind)
			
		sents_emb = [] 
		for bound_sent in bound_sents:
			sents_emb.append(self.pool_sequential_embed(tok_embed , bound_sent[0] , bound_sent[1] , method_pool) )
		sents_emb = torch.stack(sents_emb , dim = 0) #shape (N_sent , sent dim )

		return sents_emb

	def pool_sequential_embed(self, roberta_embed , start , end , method):

		if method =='mean':
			sub_matrix = roberta_embed[start:end+1 , :] 
			return torch.mean(sub_matrix , axis = 0 ) 
		elif method == 'att':
			#func_emb (N_func , word dim) 
			#using one attention layer to compute the document embedding
			sub_matrix = roberta_embed[start:end+1 , :] 

			func_att = self.W_q_sent(sub_matrix) #shape (N func , 1)
			func_val = self.W_v_sent(sub_matrix) #shape (N func , word dim)

			attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_func , 1 )
			attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N func )
			result = torch.matmul(attention_matrix ,  func_val) #shape (1 , word_dim)
			return result.squeeze(0)

def find_boudary_sents(tok2sent_idx):
	#bound sents
	sent_tok = {}

	for i in range(len(tok2sent_idx)):
		if tok2sent_idx[i] not in sent_tok:
			sent_tok[tok2sent_idx[i]] = [i]
		else:
			sent_tok[tok2sent_idx[i]].append(i)

	bound_sents = [] 
	for sent_id in sent_tok:
		bound_sents.append([sent_tok[sent_id][0],sent_tok[sent_id][-1]])

	return bound_sents
