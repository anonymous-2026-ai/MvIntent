from os import listdir
from os.path import isdir, join
import os

global_path = "./current_save"

best_macro = -1.0
current_path = ""
current_file = ""
for x in listdir(global_path):
    sub_dir = join(global_path, x)

    if not isdir(sub_dir) or '.' not in x:
        continue

    # duyệt tất cả file trong sub_dir
    for fname in listdir(sub_dir):
        if fname.startswith("test_results") and fname.endswith(".txt"):
            file_path = join(sub_dir, fname)

            with open(file_path, "r") as f:
                lines = [l.strip() for l in f.readlines()]

            # giả sử macro nằm ở dòng đầu
            macro_line = lines[0]
            macro_score = float(macro_line.split("=")[1].strip())
            # print('macro score ' , macro_score)
            if macro_score > 0.8:
                best_macro = macro_score
                current_path = sub_dir
                current_file = file_path
                print('best macro ' , best_macro , 'current path', current_path)
print("Best run:", current_path)
print("Best macro:", best_macro)
print("Curent file ", current_file)
