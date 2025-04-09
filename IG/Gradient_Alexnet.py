import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2

# ----------------------------
# Load the trained AlexNet model
# ----------------------------
model_path = './FF++/Models/Alexnet_model.pth'
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  # Binary classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ----------------------------
# Define preprocessing for AlexNet
# ----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

# ----------------------------
# Generate gradient saliency map
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
    print(f"Saliency map saved at {output_path}")

# ----------------------------
# Only use these specific subfolders
# ----------------------------
input_root = './FF++/test'
selected_folders = ['original_sequences_actors', 'original_sequences_youtube', 'manipulated_sequences']
output_root = './FF++/test/IG/Alexnet'

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
