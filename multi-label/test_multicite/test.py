# import torch
# from torch.nn.utils.rnn import pad_sequence
# from transformers import AutoTokenizer, AutoModel

# SCIBERT = "allenai/scibert_scivocab_uncased"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1) Load tokenizer + model (để lấy pad_token_id chuẩn và chạy SciBERT)
# tokenizer = AutoTokenizer.from_pretrained(SCIBERT)
# model = AutoModel.from_pretrained(SCIBERT).to(device)
# model.eval()

# pad_id = tokenizer.pad_token_id  # thường là 0 với BERT/SciBERT
# print(pad_id)
from transformers import AutoTokenizer

# def scibert_tokenize(text: str):
#     """
#     Tokenize a string using SciBERT tokenizer.
#     Output: list of tokens (WordPiece)
#     """
#     tokenizer = AutoTokenizer.from_pretrained(
#         "allenai/scibert_scivocab_uncased"
#     )

#     tokens = tokenizer.tokenize(text)
#     return tokens


if __name__ == "__main__":
    # sentence = "This model improves citation intent detection."

    # tokens = scibert_tokenize(sentence)

    # for t in tokens:
    #     print(t)
    l = [ [0, 1] , [2,1]]
    print( [0,1] in l)


