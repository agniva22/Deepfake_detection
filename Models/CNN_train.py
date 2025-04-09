import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

# Custom CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        real_folders = ['Celeb-real', 'YouTube-real']
        fake_folders = ['Celeb-synthesis']

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

        print(f" Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image)  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

root_folder = './Celeb-DF/train' 
dataset = FaceDataset(root_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = CNNModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_losses = []
train_accuracies = []

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
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save the trained model
model_save_path = '/./Celeb-DF/Model/cnn_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot and save loss/accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Loss", color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label="Accuracy", color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plot_save_path = './Celeb-DF/cnn_training_curves.png'
plt.savefig(plot_save_path)
print(f"📊 Training curves saved to {plot_save_path}")
plt.close()
