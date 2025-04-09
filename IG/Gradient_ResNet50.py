import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import os
import cv2
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load fine-tuned ResNet50
model_path = './FF++/Models/ResNet50_model.pth'
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Saliency map generator
def generate_saliency_map(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(device)
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
    print(f"Saved: {output_path}")

input_root = './FF++/test'
selected_folders = ['original_sequences_actors', 'original_sequences_youtube', 'manipulated_sequences']
output_root = './FF++/test/IG/ResNet50'

for folder_name in selected_folders:
    folder_path = os.path.join(input_root, folder_name)
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, fname)
                rel_path = os.path.relpath(image_path, input_root)
                output_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '_gradient.png')
                generate_saliency_map(image_path, output_path)
