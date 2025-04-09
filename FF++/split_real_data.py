import os
import shutil
import random
from math import ceil

# Source dataset folders
source_dirs = {
    'original_sequences_actors': '/home/ant-pc/Downloads/FF++/original_sequences_actors',
    'original_sequences_youtube': '/home/ant-pc/Downloads/FF++/original_sequences_youtube'
}

destination_root = '/home/ant-pc/papers/FG/FF++'
train_dir = os.path.join(destination_root, 'train')
val_dir = os.path.join(destination_root, 'val')
test_dir = os.path.join(destination_root, 'test')

def copy_folder(src, dst):
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

train_ratio = 0.72
val_ratio = 0.14
test_ratio = 0.14

for category, folder in source_dirs.items():
    if not os.path.exists(folder):
        print(f"[SKIP] Missing: {category}")
        continue

    print(f"[PROCESS] Category: {category}")

    subfolders = [sub for sub in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub))]
    random.shuffle(subfolders)

    total = len(subfolders)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_subfolders = subfolders[:train_end]
    val_subfolders = subfolders[train_end:val_end]
    test_subfolders = subfolders[val_end:]

    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    test_category_dir = os.path.join(test_dir, category)

    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(val_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    for subfolder in train_subfolders:
        print(f"  [TRAIN] {subfolder}")
        copy_folder(os.path.join(folder, subfolder), os.path.join(train_category_dir, subfolder))

    for subfolder in val_subfolders:
        print(f"  [VAL] {subfolder}")
        copy_folder(os.path.join(folder, subfolder), os.path.join(val_category_dir, subfolder))

    for subfolder in test_subfolders:
        print(f"  [TEST] {subfolder}")
        copy_folder(os.path.join(folder, subfolder), os.path.join(test_category_dir, subfolder))

print("\nDataset successfully split into train (72%), val (14%), and test (14%).")
