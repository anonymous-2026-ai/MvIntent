import spacy
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
nlp = spacy.load("en_core_web_sm")
with open('ner_example.txt' ,'r') as f_r:
	text = f_r.read()

print('text ' , text)

doc = nlp(text)
tokens1 = tokenizer.tokenize(text)

def find_indexes(noun_chunk_token , context_token, index_return):

	indices = []

	for i in range(len(context_token) - len(noun_chunk_token) + 1):
		if context_token[i:i+len(noun_chunk_token)] == noun_chunk_token:
			indices.append(list(range(i, i+len(noun_chunk_token))))
	
	if len(indices) > 0:
		return indices[index_return]
	else:
		return []
	
def count_previous_duplicates_lists(lst):
	seen = {}
	result = []

	for x in lst:
		key = tuple(x)   # chuyển list → tuple để làm key
		count = seen.get(key, 0)
		result.append(count)
		seen[key] = count + 1

	return result
	
print('tokens1', tokens1, len(tokens1))
noun_chunks = [chunk.text for chunk in doc.noun_chunks]
# noun_chunk_tokens  = [ tokenizer.tokenize(chunk)  for chunk in noun_chunks]
# noun_chunk_appear = count_previous_duplicates_lists(noun_chunk_tokens)

# for i in range(len(noun_chunk_tokens)):
# 	chunk_tokens , index_return = noun_chunk_tokens[i] , noun_chunk_appear[i]
# 	list_chunk_indexes  = find_indexes(chunk_tokens , tokens1, index_return)
# 	print(chunk_tokens , list_chunk_indexes)
	
print(noun_chunks)