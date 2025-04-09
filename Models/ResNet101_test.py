import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datetime import datetime
import numpy as np

# Custom dataset for nested folder structure
class CelebDFTestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        real_folders = ['Celeb-real', 'YouTube-real']
        fake_folders = ['Celeb-synthesis']

        for folder in real_folders:
            class_dir = os.path.join(folder_path, folder)
            if os.path.exists(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(0)

        for folder in fake_folders:
            class_dir = os.path.join(folder_path, folder)
            if os.path.exists(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(1)

        print(f"Loaded {len(self.image_paths)} test images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Paths
test_folder = './Celeb-DF/test'
model_path = './Celeb-DF/Models/ResNet101_model.pth'
results_file = './Celeb-DF/Result/ResNet101_test_results.txt'

# DataLoader
test_dataset = CelebDFTestDataset(test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of test images: {len(test_dataset)}")

# Load model
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluation
y_true = []
y_pred = []
y_pred_probs = []  # This will store the predicted probabilities for ROC AUC

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Get predicted class labels
        _, predicted = torch.max(outputs, 1)

        # Get predicted probabilities for ROC AUC (positive class)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_pred_probs.extend(probabilities)

# Compute metrics
acc = accuracy_score(y_true, y_pred) * 100
prec = precision_score(y_true, y_pred, average='binary') * 100
rec = recall_score(y_true, y_pred, average='binary') * 100
f1 = f1_score(y_true, y_pred, average='binary') * 100
cm = confusion_matrix(y_true, y_pred)

# Compute ROC AUC score
roc_auc = roc_auc_score(y_true, y_pred_probs) * 100

# Format results
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
result_str = f"""
--- ResNet101 Test Results ({timestamp}) ---
Total Images Tested: {len(test_dataset)}

Test Accuracy : {acc:.2f}%
Precision      : {prec:.2f}%
Recall         : {rec:.2f}%
F1-Score       : {f1:.2f}%
ROC AUC        : {roc_auc:.2f}%

Confusion Matrix:
{cm}
"""

print(result_str)

# Save to file
with open(results_file, 'w') as f:
    f.write(result_str)

print(f"All metrics saved to: {results_file}")
