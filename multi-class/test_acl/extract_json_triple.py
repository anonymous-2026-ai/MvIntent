import json 
import sys 
with open('data_acl_arc/test_triple.json','r') as f:
    data = json.load(f)
    index = int(sys.argv[1])
    print(data[index])