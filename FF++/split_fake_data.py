import os
import shutil
import json
import random

# Source dataset folder
source_dir = '/home/ant-pc/papers/FG/FF++/manipulated_sequences'

destination_root = '/home/ant-pc/papers/FG/FF++'
train_dir = os.path.join(destination_root, 'train', 'manipulated_sequences')
val_dir = os.path.join(destination_root, 'val', 'manipulated_sequences')
test_dir = os.path.join(destination_root, 'test', 'manipulated_sequences')

train_json_path = '/home/ant-pc/papers/FG/FF++/train.json'
val_json_path = '/home/ant-pc/papers/FG/FF++/val.json'
test_json_path = '/home/ant-pc/papers/FG/FF++/test.json'


def copy_folder(src, dst):
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

def load_folder_names(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return [f"{item[0]}_{item[1]}" for item in data]

train_folders = load_folder_names(train_json_path)
val_folders = load_folder_names(val_json_path)
test_folders = load_folder_names(test_json_path)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


all_folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]
random.shuffle(all_folders)

for folder in all_folders:
    src_path = os.path.join(source_dir, folder)
    if folder in train_folders:
        dst_path = os.path.join(train_dir, folder)
        print(f"  [TRAIN] {folder}")
        copy_folder(src_path, dst_path)
    elif folder in val_folders:
        dst_path = os.path.join(val_dir, folder)
        print(f"  [VAL] {folder}")
        copy_folder(src_path, dst_path)
    elif folder in test_folders:
        dst_path = os.path.join(test_dir, folder)
        print(f"  [TEST] {folder}")
        copy_folder(src_path, dst_path)
