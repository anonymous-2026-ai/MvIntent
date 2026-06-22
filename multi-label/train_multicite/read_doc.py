import json 
import torch 
import sys

index = int(sys.argv[1])
with open('data_multicite_triple/test.json' , 'r') as f:
    all_data = json.load(f)
    print(all_data[index]['list_triple'])
    print(all_data[index]['x'])