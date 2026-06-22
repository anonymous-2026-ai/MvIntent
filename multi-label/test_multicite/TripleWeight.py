from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class TripleWeight(nn.Module):

	def __init__(self,  word_emb_dim, ):
		
		super().__init__()
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.W_d = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.doc_context_vector = nn.Parameter(torch.randn(self.word_emb_dim))

	def forward(self, triple_embeddings):
		query = self.doc_context_vector
		func_key = self.W_d(triple_embeddings) #shape (N,   dim)
		z1 = ( torch.matmul( query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func ) 
		z1 = 1. / math.sqrt(self.word_emb_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z1) #shape (N_triple )
		return torch.matmul(attention_matrix , triple_embeddings ) , attention_matrix.tolist() # shape (768) , 