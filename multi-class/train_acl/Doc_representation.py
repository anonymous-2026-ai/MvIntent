from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class Doc_representation(nn.Module):

	def __init__(self,  word_emb_dim, n_func , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		# self.W_q = nn.ModuleList([nn.Linear(self.word_emb_dim , self.word_emb_dim) for i in range(n_func)]) # query matrix
		# # self.W_v = nn.ModuleList([nn.Linear(self.word_emb_dim , self.word_emb_dim) for i in range(n_func)])
		# self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim)

		# self.W_k = nn.ModuleList([nn.Linear(self.word_emb_dim , self.word_emb_dim) for i in range(n_func)])
		self.W_q = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_k = nn.Linear(self.word_emb_dim , self.word_emb_dim)


		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		# self.W_q = nn.Linear(self.word_emb_dim , 1) # query matrix
		self.doc_linear = nn.Linear(self.word_emb_dim , n_func)

		# self.W_q = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix
		# self.W_k = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix
		# self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix

		self.W_q_doc = nn.Linear(self.word_emb_dim , 1) # value matrix
		self.W_v_doc = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix

	# def forward(self, func_embed):
	# 	#func_emb (N_func , word dim) 
	# 	#using one attention layer to compute the document embedding
	# 	func_att = self.W_q_doc(func_embed) #shape (N func , 1)
	# 	func_val = self.W_v_doc(func_embed) #shape (N func , word dim)

	# 	attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_func , 1 )
	# 	attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N func )
	# 	result = torch.matmul(attention_matrix ,  func_val) #shape (1 , word_dim)
	# 	return result.squeeze(0), attention_matrix


	def forward(self, sent_emb):
		return torch.mean(sent_emb , dim = 0)
	
	def compute_doc_logit(self, doc_emb):
		#doc emb (word)
		# doc_logit = []
		# for i in range(len(self.doc_linear)):
		# 	doc_logit.append(self.doc_linear[i](doc_emb))
		# return doc_logit #list of n_func element
		# doc_repeat = doc_emb.repeat(func_hid.shape[0] , 1)
		# doc_repeat = doc_repeat.to(self.device)
		# return self.doc_linear(torch.cat([doc_repeat , func_hid] , dim = 1 ))
		return self.doc_linear(doc_emb)
	
	# def compute_topics(self, toks_emb , num_fun):
	# 	#toks emb shape ( num tok ,  word dim )
	# 	out = [] 
	# 	list_attention_matrix = []
	# 	for i in range(len(self.w_linear)):
	# 		tok_hid = self.w_linear[i](toks_emb) #shape (n_tok , tok dim)
	# 		func_att = self.w_att[i](tok_hid) #shape (n tok , 1 )
	# 		attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_tok , 1 )
	# 		list_attention_matrix.append(attention_matrix.squeeze(1))
	# 		attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N tok )
	# 		result = torch.matmul(attention_matrix ,  tok_hid).squeeze(0) #shape (word_dim))
	# 		out.append(result)
	# 	out = torch.stack(out)
	# 	return out , list_attention_matrix
	
	# def compute_topics(self, toks_emb , num_fun):
	# 	#toks emb shape ( num tok ,  word dim )
	# 	out = [] 
	# 	list_attention_matrix = []
	# 	for i in range(len(self.w_linear)):
	# 		# tok_hid = self.w_linear[0](toks_emb) #shape (n_tok , tok dim)
	# 		func_att = self.w_att[i](toks_emb) #shape (n tok , 1 )
	# 		attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_tok , 1 )
	# 		list_attention_matrix.append(attention_matrix.squeeze(1))
	# 		attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N tok )
	# 		result = torch.matmul(attention_matrix ,  toks_emb).squeeze(0) #shape (word_dim))
	# 		out.append(result)
	# 	out = torch.stack(out)
	# 	return out , list_attention_matrix
		
	def compute_topics_v2(self, toks_embed , init_emb, return_score = False):

		sent_query = self.W_q(init_emb) #shape (N_sent , sent dim)
		func_key = self.W_k(toks_embed) #shape (N func, word dim)
		func_val = self.W_v(toks_embed) #shape (N func , word dim)
		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		z = 1. / math.sqrt(self.word_emb_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
		result = init_emb + result
		if return_score == False:
			return result #shape (N_sent, sent dim)
		else:
			#return score = True, return score of attention weight 
			return result,  attention_matrix


	def compute_topics_v3(self, toks_embed , init_emb, return_score = False):
		num_func = init_emb.shape[0]
		#toks emb shape ( num tok ,  word dim )
		out = [] 
		list_attention_matrix = []
		for i in range(num_func):
			sent_query = self.W_q[i](init_emb[i, :].unsqueeze(0)) #shape (1 , tok dim)
			tok_key = self.W_k[i](toks_embed) #shape (N -tok, word dim)
			tok_val = self.W_v(toks_embed) #shape (N tok, word dim)
			z1 = ( torch.matmul( sent_query , torch.transpose(tok_key , 0 , 1 )) ) # shape (1 , N tok )
			z = 1. / math.sqrt(self.word_emb_dim) * z1
			attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (1 , N_tok )
			list_attention_matrix.append(attention_matrix.tolist())
			result = torch.matmul(attention_matrix ,  tok_val).squeeze(0) #shape (word dim))
			out.append(result)
		out = torch.stack(out)
		return out , list_attention_matrix

