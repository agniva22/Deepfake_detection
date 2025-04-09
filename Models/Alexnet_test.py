import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import cv2
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

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
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Paths
test_folder =  './Celeb-DF/test/IG/Alexnet'
model_path = './Models/Alexnet_model.pth'
results_file = './Celeb-DF/Result/alexnet_test_results.txt'

# Dataset and dataloader
test_dataset = CelebDFTestDataset(test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load AlexNet
model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Metrics
TP = FP = TN = FN = 0
all_labels = []
all_preds = []
all_probs = []  

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        TP += ((predicted == 1) & (labels == 1)).sum().item()
        TN += ((predicted == 0) & (labels == 0)).sum().item()
        FP += ((predicted == 1) & (labels == 0)).sum().item()
        FN += ((predicted == 0) & (labels == 1)).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        probs = torch.nn.Softmax(dim=1)(outputs)
        all_probs.extend(probs.cpu().numpy()[:, 1])  

# Compute metrics
accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
results = f"""
--- AlexNet Test Results ({timestamp}) ---
Total Images Tested: {len(test_dataset)}
Accuracy: {accuracy:.2f}%
F1-Score: {f1:.2f}
Precision: {precision:.2f}
Recall (Sensitivity): {recall:.2f}
Specificity: {specificity:.2f}
True Positives: {TP}
False Positives: {FP}
True Negatives: {TN}
False Negatives: {FN}
ROC AUC: {roc_auc:.2f}
"""

print(results)

# Save results to file
with open(results_file, 'w') as f:
    f.write(results)

print(f"Results saved to: {results_file}")
