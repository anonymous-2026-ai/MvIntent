from os import listdir
from os.path import isfile , isdir
import os , json 
global_path = "./"

list_dir  = listdir(global_path)
best_macro = 0 
current_path = ''
for x in list_dir: 
    sub_dir  = os.path.join(global_path , x)
    if isdir(sub_dir) and '.' in x:
        if os.path.exists(os.path.join(sub_dir, 'global_step_best.txt')):
            global_step_best_path = os.path.join(sub_dir, 'global_step_best.txt')
            with open(global_step_best_path , 'r') as file :
                best_step = file.read().strip()

            setting_path  = os.path.join(sub_dir , 'setting.json')
            with open(setting_path ,'r') as f_r:
                setting_detail = json.load(f_r)
                if setting_detail['seed'] == 0:
                    test_result_path = os.path.join(sub_dir, 'test_results'+best_step+'.txt')
                    with open(test_result_path , 'r') as file:
                        result_lines =  [t.strip() for t in  file.readlines() ] 
                        macro_line = result_lines[1]
                        print('macro line ' , macro_line, result_lines[2], sub_dir)
                        macro_score = float(macro_line.split('=')[1].strip())
                        if macro_score >= best_macro :
                            best_macro = macro_score 
                            current_path = sub_dir

print(current_path)
