import os
import shutil

# Source dataset folders
source_dirs = {
    'Celeb-real': '/home/ant-pc/papers/FG/Celeb-DF/Celeb-real',
    'Celeb-synthesis': '/home/ant-pc/papers/FG/Celeb-DF/Celeb-synthesis',
    'YouTube-real': '/home/ant-pc/papers/FG/Celeb-DF/YouTube-real'
}

# Path to test list
test_list_file = '/home/ant-pc/papers/FG/Celeb-DF/List_of_testing_videos.txt'

# Destination
destination_root = '/home/ant-pc/papers/FG/Celeb-DF'
train_dir = os.path.join(destination_root, 'train')
test_dir = os.path.join(destination_root, 'test')

test_videos = set()
with open(test_list_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        category_path, filename = parts[1].split('/')
        video_name = os.path.splitext(filename)[0]
        test_videos.add((category_path, video_name))

print(f"Loaded {len(test_videos)} test video entries.")

def copy_video(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

for category, folder in source_dirs.items():
    if not os.path.exists(folder):
        print(f"[SKIP] Missing: {category}")
        continue

    print(f"[PROCESS] Category: {category}")
    for video in os.listdir(folder):
        video = video.strip()
        video_path = os.path.join(folder, video)

        if not os.path.isdir(video_path):
            continue

        is_test = (category, video) in test_videos
        dest_base = test_dir if is_test else train_dir
        dest_path = os.path.join(dest_base, category, video)

        print(f"  [{'TEST' if is_test else 'TRAIN'}] {video}")
        copy_video(video_path, dest_path)

print("Dataset successfully split.")
