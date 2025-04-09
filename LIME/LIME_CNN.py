import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import functional as F_transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import slic
import torch.nn.functional as F

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

model_path = './Celeb-DF/Models/cnn_model.pth'
model = CNNModel(num_classes=2)
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def preprocess_image(image):
    image_tensor = F_transforms.resize(image, (128, 128))
    image_tensor = F_transforms.to_tensor(image_tensor).to(device)
    image_tensor = F_transforms.normalize(
        image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return image_tensor

explainer = lime_image.LimeImageExplainer()

def model_predict(images):
    images = torch.stack([preprocess_image(Image.fromarray(img).convert("RGB")) for img in images])
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

def segment_image(image):
    return slic(image, n_segments=100, compactness=10)

input_root = './Celeb-DF/test'
output_root = './Celeb-DF/test/LIME/CNN'

for root, _, files in os.walk(input_root):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, fname)
            rel_path = os.path.relpath(image_path, input_root)  
            output_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + '_lime.png')

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            image = Image.open(image_path).convert("RGB")
            original_image = np.array(image)

            explanation = explainer.explain_instance(
                original_image,
                model_predict,
                top_labels=2,
                hide_color=0,
                num_samples=100,
                segmentation_fn=segment_image
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=False
            )

            combined_image = np.ones_like(original_image) * 255
            combined_image[mask > 0] = original_image[mask > 0]

            Image.fromarray(combined_image.astype(np.uint8)).save(output_path)
            print(f"LIME explanation saved to {output_path}")
