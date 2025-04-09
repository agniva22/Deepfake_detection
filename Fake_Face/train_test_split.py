import os
import shutil
import random

# Path to the original data
source_dir = 'cropped_faces_output'

# Create train and test directories inside the source directory
train_dir = os.path.join(source_dir, 'train')
test_dir = os.path.join(source_dir, 'test')

# List of subfolders (categories)
categories = ['real', 'easy', 'mid', 'hard']

# Create train and test directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for category in categories:
    # Paths for the current category
    category_path = os.path.join(source_dir, category)
    
    # Get list of all files in the category folder
    files = os.listdir(category_path)
    
    # Shuffle files to ensure random split
    random.shuffle(files)
    
    # Split files into 70% for training and 30% for testing
    split_index = int(0.7 * len(files))
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    # Create category directories under train and test
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    
    # Move the files to train and test directories
    for file in train_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(train_dir, category, file))
    
    for file in test_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(test_dir, category, file))

print("Data has been successfully split into train and test directories inside 'cropped_faces_output'.")
