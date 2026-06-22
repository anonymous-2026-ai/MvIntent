from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class Word_Function_Interaction(nn.Module):

	def __init__(self,  word_emb_dim, n_iter , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.W_q_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_v_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_k_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)

		self.W_q_f2t = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_v_f2t = nn.Linear(self.word_emb_dim , self.word_emb_dim) 
		self.W_k_f2t = nn.Linear(self.word_emb_dim , self.word_emb_dim)

		self.n_iter = n_iter


	def forward(self, toks_emb , func_embed):
		#sents_emb (N_sent , word dim) 
		#func emb (N_func, word dim)
		toks_hid  = toks_emb 
		func_hid = func_embed
		#similar to that of sent, dont need to change the code with toks emb 
		for i in range(self.n_iter):
			# toks_hid = self.cross_att_f2t(func_hid , toks_hid)
			func_hid = self.cross_att_t2f(toks_hid , func_hid) # N func , word dim
		return  toks_hid , func_hid
	
	def cross_att_t2f(self, toks_embed , init_emb, return_score = False):

		sent_query = self.W_q_t2f(init_emb) #shape (N_sent , sent dim)
		func_key = self.W_q_t2f(toks_embed) #shape (N func, word dim)
		func_val = self.W_v_t2f(toks_embed) #shape (N func , word dim)

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
	
	# def cross_att_f2t(self, toks_embed , init_emb, return_score = False):

	# 	sent_query = self.W_q_f2t(init_emb) #shape (N_sent , sent dim)
	# 	func_key = self.W_k_f2t(toks_embed) #shape (N func, word dim)
	# 	func_val = self.W_v_f2t(toks_embed) #shape (N func , word dim)
	# 	z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
	# 	z = 1. / math.sqrt(self.word_emb_dim) * z1
	# 	attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
	# 	result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
	# 	result = init_emb + result
	# 	if return_score == False:
	# 		return result #shape (N_sent, sent dim)
	# 	else:
	# 		#return score = True, return score of attention weight 
	# 		return result,  attention_matrix
	
class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, func_embed , sent_embed, return_score = False):

		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
		func_key = self.W_k(func_embed) #shape (N func, word dim)
		func_val = self.W_v(func_embed) #shape (N func , word dim)
		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		z = 1. / math.sqrt(self.sent_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )

		if return_score == False:
			return result #shape (N_sent, sent dim)
		else:
			#return score = True, return score of attention weight 
			return result,  z.tolist()
