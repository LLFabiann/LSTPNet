from glob import glob
import os

your_dataset_path = "D:/DL_Study/Facial_Ex_Data/DFEW/"

all_txt_file = glob(os.path.join('DFEW*.txt'))

def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file, "/home/fabian/BRL/LCC/DL_Study/Expression_Data/DFEW/", your_dataset_path)
