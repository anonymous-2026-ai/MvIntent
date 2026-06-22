from transformers import BertModel
import torch.nn as nn
import torch 
import math 


class TripleAttention(nn.Module):

	def __init__(self, model, word_emb_dim, n_iter,device):
		
		super().__init__()
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.n_iter = n_iter
		self.cross_att = CrossAttentionBlock(self.word_emb_dim)
		self.model = model 
		self.device = device

	# def forward(self, toks_embeddings):
	# 	# print('one hot matrix ' , one_hot_matrix)
	# 	triple_hidden_states = toks_embeddings
	# 	for i in range(self.n_iter):
	# 		triple_out = self.cross_att(triple_hidden_states)
	# 		triple_hidden_states = triple_hidden_states + triple_out #residual connection 
		
	# 	return triple_hidden_states

	def forward(self, toks_embeddings):
		toks_embeddings = toks_embeddings.unsqueeze(0) # ( 1, len, 768)
		seq_len = toks_embeddings.size(1)
		attention_mask = torch.ones(1, seq_len)

		# ==============================
		# 4. Forward qua SciBERT
		# ==============================
		outputs = self.model(
			inputs_embeds=toks_embeddings.to(self.device),
			attention_mask=attention_mask.to(self.device)
		)
		last_hidden_state = outputs.last_hidden_state.squeeze(0) #shape (len , 768 )
		return last_hidden_state

class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, triple_embeddings):
		sent_query = triple_embeddings #shape (N ,  dim)
		func_key = triple_embeddings #shape (N,   dim)
		func_val = self.W_v(triple_embeddings) #shape (N func , word dim)
		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		z = 1. / math.sqrt(self.sent_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
		return result 

