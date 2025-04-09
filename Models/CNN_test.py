import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CustomTestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.real_folders = ['Celeb-real', 'YouTube-real']  
        self.fake_folders = ['Celeb-synthesis']  
        self.transform = transform

        for folder in self.real_folders:
            class_dir = os.path.join(folder_path, folder)
            if os.path.exists(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(0) 

        # Add fake class images
        for folder in self.fake_folders:
            class_dir = os.path.join(folder_path, folder)
            if os.path.exists(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(1)  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# Paths
test_folder = './Celeb-DF/test' 
model_path = './Celeb-DF/Models/cnn_model.pth'  
results_file = './Celeb-DF/Result/cnn_test_results.txt'  

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Dataset and dataloader
test_dataset = CustomTestDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Number of test samples: {len(test_dataset)}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = CNNModel(num_classes=2) 
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

y_true, y_pred, y_pred_probs = [], [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        
        probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_pred_probs.extend(probabilities)

# Calculate metrics
test_accuracy = accuracy_score(y_true, y_pred) * 100
test_f1_score = f1_score(y_true, y_pred, average='binary') * 100
roc_auc = roc_auc_score(y_true, y_pred_probs) * 100

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
result_str = f"""
--- CNN Test Results ({timestamp}) ---
Total Images Tested: {len(test_dataset)}
Test Accuracy: {test_accuracy:.2f}%
Test F1-score: {test_f1_score:.2f}%
Test ROC AUC: {roc_auc:.2f}%
"""

print(result_str)

# Save to file
with open(results_file, 'w') as f:
    f.write(result_str)

print(f"Results saved to: {results_file}")
