import torch
from torch.nn.utils.rnn import pad_sequence

# 1. Giả lập một batch gồm 3 câu có độ dài khác nhau (Variable-length sequences)
# Các con số đại diện cho ID của các từ (token IDs) sau khi qua Tokenizer
seq_1 = torch.tensor([45, 12, 89, 23])          # Độ dài = 4
seq_2 = torch.tensor([11, 99, 54, 32, 77, 88])  # Độ dài = 6 (Dài nhất trong batch -> Lmax)
seq_3 = torch.tensor([5, 10])                   # Độ dài = 2

# Gom các câu này lại thành một danh sách (List of Tensors)
seqs = [seq_1, seq_2, seq_3]

# Định nghĩa ID dùng để đệm, thông thường trong BERT/SciBERT là 0
pad_id = 0


# 3. Chạy hàm pad_sequence
input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, Lmax)

print(seqs)
print(input_ids)