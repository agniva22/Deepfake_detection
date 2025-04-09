import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        real_folders = ['original_sequences_actors', 'original_sequences_youtube']
        fake_folders = ['manipulated_sequences']

        # Load real images (label 0)
        for folder_name in real_folders:
            full_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(full_path):
                for subfolder in os.listdir(full_path):
                    subfolder_path = os.path.join(full_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file in os.listdir(subfolder_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                self.image_paths.append(os.path.join(subfolder_path, file))
                                self.labels.append(0)

        # Load fake images (label 1)
        for folder_name in fake_folders:
            full_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(full_path):
                for subfolder in os.listdir(full_path):
                    subfolder_path = os.path.join(full_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file in os.listdir(subfolder_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                self.image_paths.append(os.path.join(subfolder_path, file))
                                self.labels.append(1)

        print(f"✅ Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Path to training data
train_root = '/home/ant-pc/papers/FG/FF++/train'

# Load dataset
train_dataset = FaceDataset(train_root, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Load AlexNet
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100
    train_losses.append(epoch_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

# Save model
save_path = '/home/ant-pc/papers/FG/FF++/Model/Alexnet_model.pth'
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")

# Save loss curve
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve (AlexNet)')
loss_path = '/home/ant-pc/papers/FG/FF++/training_loss_curve.png'
plt.savefig(loss_path)
print(f"📈 Loss curve saved to {loss_path}")
plt.close()
