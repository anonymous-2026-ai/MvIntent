from transformers import BertModel
import torch.nn as nn
import torch 
import math 


def flatten_recursive(lst):
    for item in lst:
        if isinstance(item, list):
            # nếu item là list → flatten tiếp
            yield from flatten_recursive(item)
        else:
            # nếu là phần tử cuối → trả về
            yield item

class TripleEnhanceLayer(nn.Module):

	def __init__(self,  word_emb_dim, n_iter , alpha,  device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.n_iter = n_iter
		self.linear_pool = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.token2triple = nn.ModuleList([ CrossAttentionBlock(self.word_emb_dim , self.device) for _ in range(self.n_iter)])
		self.token2token = nn.ModuleList([CrossAttentionBlock(self.word_emb_dim, self.device) for _ in range(self.n_iter)])
		self.triple2token = nn.ModuleList([CrossAttentionBlock(self.word_emb_dim , self.device) for _ in range(self.n_iter)])
		self.intent2triple = nn.ModuleList([CrossSigmoidBlock(self.word_emb_dim) for _ in range(self.n_iter)])
		self.alpha = alpha
		# self.ln = nn.LayerNorm(self.sent_emb_dim) 

	def word_pair_to_token_pair(self, pair , tokenizer):
		source_toks  = tokenizer.encode(pair[0] , add_special_tokens=False)
		target_toks = tokenizer.encode(pair[1] , add_special_tokens=False)
		out = [] 
		for st in source_toks:
			for tt in target_toks:
				if st != tt :
					out.append([st , tt])
		return out 
	
	def word_graph_to_token_graph(self, word_graph , tokenizer):
		list_token_pair = [] 
		for source_word in word_graph:
			list_target_word = word_graph[source_word] 
			for target_word in list_target_word:
				pair = (source_word , target_word)
				list_token_pair.extend(self.word_pair_to_token_pair(pair , tokenizer))
		# list_token_pair = list(set(list_token_pair)) 

		# out= {}
		# for pair in list_token_pair:
		# 	source , target = pair[0] , pair[1]
		# 	if source not in out:
		# 		out[source] = [target]
		# 	else:
		# 		out[source].append(target)

		return list_token_pair
	
	def create_token_mask_matrix(self, chunk_index , list_token_pair):
		# print('list token pair ' , list_token_pair)
		mask_matrix = torch.zeros(len(chunk_index), len(chunk_index), dtype = torch.long)  #shape (num token , num token)
		for i in range(len(chunk_index)):
			for j in range(len(chunk_index)):
				# print('pair ' ,  [chunk_index[i] , chunk_index[j]], [chunk_index[i] , chunk_index[j]] in list_token_pair )
				if [chunk_index[i] , chunk_index[j]] in list_token_pair:
					mask_matrix[i ,j] = 0
				else:
					mask_matrix[i,j] = -1e6
		return mask_matrix

	def token_to_token(self, token_embeddings, iter, alpha):
		#shape token embedds (N_token , 768) 
		#mask matrix shape (N_token , N_token)
		return  self.token2token[iter](token_embeddings, token_embeddings, alpha = alpha)
	
	def token_to_triple(self , token_embedds, triple_embeddings , iter, alpha):
		return self.token2triple[iter](token_embedds , triple_embeddings,  alpha = alpha)
	
	def triple_to_token(self, triple_embeddings , token_embedds , iter, alpha):
		return self.triple2token[iter](triple_embeddings , token_embedds , alpha = alpha) # shape (len(list_non_index , 768))
	
	def forward(self, triple_embeddings , token_embedds, intent_embeds):
		triple_hid , token_hid  = triple_embeddings , token_embedds
		for i in range(self.n_iter):
			# token_hid = token_embedds
			triple_hid, intent_triple_sigmoid = self.intent2triple[i](intent_embeds , triple_hid, alpha = self.alpha)
			# token_hid = self.token_to_token(token_hid, i , alpha = self.alpha)
			triple_hid = self.token_to_triple(token_hid, triple_hid, i, alpha = self.alpha)
			token_hid = self.triple_to_token( triple_hid , token_hid, i, alpha = self.alpha)
		# return triple_hid , intent_triple_sigmoid
		return triple_hid , intent_triple_sigmoid


class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim , device ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix
		self.device = device 
		# self.ln = nn.LayerNorm(self.sent_dim) 

	def forward(self, tok_embeds , triple_embeddings, alpha, return_attention = False): 
		sent_query = triple_embeddings 
		func_key = tok_embeds
		func_val = self.W_v(tok_embeds) #shape (N func , word dim)
		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_triple  , N_toks)
		z1 = 1. / math.sqrt(self.sent_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z1 , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
		if return_attention == True :
			return triple_embeddings + alpha * result, attention_matrix.tolist()
		else:
			return triple_embeddings + alpha * result

class CrossSigmoidBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix
		# self.ln = nn.LayerNorm(self.sent_dim) 

	def forward(self, intent_embeddings , triple_embeddings, alpha):
		#triple embedding shape (N_triple , 768)
		triple_intent_sigmoid = torch.matmul(triple_embeddings , intent_embeddings.T)  # N_triple , N_intent 
		abs_sum = torch.sum(torch.abs(triple_intent_sigmoid) , dim = 1).unsqueeze(1) # Ntriple , 1
		triple_intent_norm = triple_intent_sigmoid / abs_sum
		intent_information = torch.matmul(triple_intent_norm , intent_embeddings) #shape (N_triple , 768)
		return triple_embeddings + alpha* intent_information , triple_intent_sigmoid.tolist() # 7, 768 


if __name__ == '__main__':
	triple = [1, 2 ,3 , 4 ]

	print(list(flatten_recursive(triple)))