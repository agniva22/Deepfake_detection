import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import slic
from PIL import Image
from matplotlib import pyplot as plt

model_path = './Celeb-DF/Models/Resnet101_Model.pth'
model = models.resnet101(pretrained=True)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).to(device)

explainer = lime_image.LimeImageExplainer()

def model_predict(images):
    images = torch.stack([preprocess_image(np.array(img)) for img in images])
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

def segment_image(image):
    return slic(image, n_segments=100, compactness=10)

image_dir = './Celeb-DF/test'
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

output_dir = './Celeb-DF/test/LIME/ResNet101'
os.makedirs(output_dir, exist_ok=True)

for image_path in image_paths:

    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)

    explanation = explainer.explain_instance(
        original_image,
        model_predict,
        top_labels=2,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segment_image,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )

    combined_image = np.ones_like(original_image) * 255
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:
                combined_image[i, j] = original_image[i, j]

    # Save the LIME mask
    image_name = os.path.basename(image_path)
    combined_output_name = os.path.splitext(image_name)[0] + '_lime.png'
    combined_output_path = os.path.join(output_dir, combined_output_name)
    Image.fromarray(combined_image.astype(np.uint8)).save(combined_output_path)
    print(f"LIME mask saved at {combined_output_path}")
