import json 

with open('data_multicite_cfu/test.json','r') as f:
    all_data = json.load(f)
    gold_labels = [ sample['y'].strip() for sample in all_data ]

with open('1780900164.9749525/predictions5200.txt' , 'r') as f:
    pred_lines = f.readlines()
    pred_lines = [ line.strip() for line in pred_lines ]

with open('check.txt','w') as f:
    for i, (gold, pred) in enumerate(zip(gold_labels , pred_lines)):
        if gold == pred:
            f.write(str(i) + ' ' + gold+ '\n')