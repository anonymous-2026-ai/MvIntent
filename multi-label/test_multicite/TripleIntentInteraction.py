from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class TripleIntentInteraction(nn.Module):

	def __init__(self,  word_emb_dim, n_iter):
		
		super().__init__()
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		# self.W_q_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.W_v_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		# self.W_k_t2f = nn.Linear(self.word_emb_dim , self.word_emb_dim)
		self.intent_embeds = nn.Parameter(torch.randn(7, 768))
		self.n_iter = n_iter

	def forward(self, triple_embeds):
		triple_hids = triple_embeds
		for i in range(self.n_iter):
			triple_enrich , attention_matrix  = self.cross_att_t2f(self.intent_embeds , triple_hids) # N func , word dim
			triple_hids = triple_hids + triple_enrich  
		return  triple_hids, attention_matrix
	
	def cross_att_t2f(self, toks_embed , init_emb):

		sent_query = init_emb #shape (N_sent , sent dim)
		func_key = toks_embed #shape (N func, word dim)
		func_val = self.W_v_t2f(toks_embed) #shape (N func , word dim)

		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		z = 1. / math.sqrt(self.word_emb_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		# print('intent shape ' , attention_matrix.shape)
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
		return result,  attention_matrix.tolist()