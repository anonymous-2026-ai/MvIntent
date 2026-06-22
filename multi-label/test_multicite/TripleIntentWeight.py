from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class TripleIntentWeight(nn.Module):

	def __init__(self,  word_emb_dim, n_iter):
		
		super().__init__()
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.W_d = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.W_v = nn.Linear(self.word_emb_dim , self.word_emb_dim , bias=True)
		self.doc_context_vector = nn.Parameter(torch.randn(self.word_emb_dim))
		# self.intent_embeds = nn.Parameter(torch.randn(10 , self.word_emb_dim))
		self.triple_linear = nn.Linear(768 , 768)
		self.intent_to_triple = nn.Linear(768, 768)
		self.n_iter = n_iter

	def triple_weight(self, triple_embeddings):
		query = self.doc_context_vector
		func_key = self.W_d(triple_embeddings) #shape (N,   dim)
		z1 = ( torch.matmul( query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func ) 
		z1 = 1. / math.sqrt(self.word_emb_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z1) #shape (N_triple )
		return attention_matrix 

	def triple_intent_att(self , triple_embeddings):
		#triple embedding shape (N_triple , 768 )
		# return torch.sigmoid(torch.matmul(triple_embeddings , self.intent_embeds.T)) #shape (N_triple , N_intent)
		score =  1. / math.sqrt(self.word_emb_dim) * torch.matmul(triple_embeddings , self.intent_embeds.T) #shape (N_triple , N_intent)
		atten_score = torch.nn.functional.softmax(score , dim = 1 )
		return atten_score #shape (N_triple , N_intent)
	
	# def forward(self, triple_embeddings):
	# 	eps = 1e-6
	# 	triple_weights = self.triple_weight(triple_embeddings) #shape (N_triple )
	# 	triple_intent_att = self.triple_intent_att(triple_embeddings) #shape (N_triple , N_intent )
	# 	return torch.matmul(triple_weights , triple_intent_att).clamp(eps, 1 - eps) , triple_weights.tolist() , triple_intent_att.tolist() #shape (N_intent )

	# def forward(self, triple_embeddings, word_vector):
	# 	eps = 1e-6
	# 	triple_weights = self.triple_weight(triple_embeddings) #shape (N_triple )
	# 	overall_triple = torch.matmul(triple_weights , self.triple_linear(triple_embeddings)) + word_vector #shape (768) 
	# 	intent_scores = self.intent_linears(overall_triple) #shape (N_intent)
	# 	# return intent_scores , triple_weights.tolist() , torch.sigmoid(self.intent_linears(triple_embeddings)).tolist() #shape (N_intent )
	# 	# if triple_embeddings.shape[0] == 2:
	# 	# 	print(triple_embeddings[:, : 8])
	# 	return intent_scores , triple_weights.tolist() , torch.sigmoid(self.intent_linears(triple_embeddings)).tolist() #shape (N_intent )

	def intent_to_triple(self, triple_embeddings, intent_linear):
		#triple embedding shape (N_triple , 768)
		intent_embeddings = intent_linear.weight #shape (7,768)
		triple_intent_sigmoid = torch.sigmoid(intent_linear(triple_embeddings))  # N_triple , N_intent 
		triple_intent_sigmoid  = triple_intent_sigmoid / torch.sum(triple_intent_sigmoid , dim = 1, keepdim = True )
		intent_information = torch.matmul(triple_intent_sigmoid , intent_embeddings) #shape (N_triple , 768)
		return triple_embeddings + intent_information , triple_intent_sigmoid.tolist() # 7, 768 

	def forward(self, triple_embeddings, intent_linear):
		#triple embedding shape (N_triple , 768)
		triple_hid = triple_embeddings
		for _ in range(self.n_iter):
			triple_hid , triple_intent_sigmoid = self.intent_to_triple(triple_hid, intent_linear)
		return triple_hid , triple_intent_sigmoid