import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import functional as F_transforms
from PIL import Image
import torch.nn.functional as F
import cv2

# ----------------------------
# Model Definition
# ----------------------------
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
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Load Model
# ----------------------------
model_path = './FF++/Models/cnn_model.pth'
model = CNNModel(num_classes=2)
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_image(image):
    image_tensor = F_transforms.resize(image, (128, 128))
    image_tensor = F_transforms.to_tensor(image_tensor).to(device)
    image_tensor = F_transforms.normalize(
        image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return image_tensor.unsqueeze(0)

# ----------------------------
# Saliency Map Generator
# ----------------------------
def generate_saliency_map(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    input_tensor = preprocess_image(image)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    class_idx = torch.argmax(output).item()

    model.zero_grad()
    output[0, class_idx].backward()

    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = (saliency * 255).astype(np.uint8)

    saliency_resized = cv2.resize(saliency, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(saliency_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Gradient saliency map saved to {output_path}")

# ----------------------------
# Walk Through Dataset Folder
# ----------------------------
input_root = './FF++/test'
selected_folders = ['original_sequences_actors', 'original_sequences_youtube', 'manipulated_sequences']
output_root = './FF++/test/IG/CNN'

# ----------------------------
# Traverse and process
# ----------------------------
for folder_name in selected_folders:
    folder_path = os.path.join(input_root, folder_name)
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, fname)
                rel_path = os.path.relpath(image_path, input_root)
                output_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '_gradient.png')
                generate_saliency_map(image_path, output_path)
