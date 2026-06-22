from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class TripleEnrichLayer(nn.Module):

	def __init__(self,  word_emb_dim , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.W_d = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.doc_context_vector = nn.Parameter(torch.randn(self.word_emb_dim))

	def forward(self, cls_embedding, triple_embeddings):
		query = self.doc_context_vector
		# func_key = torch.tanh(self.W_d(triple_embeddings)) #shape (N,   dim)		
		func_key = self.W_d(triple_embeddings) #shape (N,   dim)
		func_val = self.W_v(triple_embeddings) #shape (N func , word dim)
		z1 = ( torch.matmul( query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func ) 
		z1 = 1. / math.sqrt(self.word_emb_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z1) #shape (N_sent , N_func )
		# print('attention matrix ' , attention_matrix.tolist())
		triple_embed_level = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )
		doc_embedding = cls_embedding  + triple_embed_level
		return doc_embedding, attention_matrix.tolist()
