import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import slic
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import vgg19, VGG19_Weights

model_path = './Celeb-DF/Model/VGG19_model.pth'
weights = VGG19_Weights.DEFAULT 
model = vgg19(weights=weights) 
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_path)) 
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).to(device)

# Define LIME explainer
explainer = lime_image.LimeImageExplainer()

def model_predict(images):
    images = torch.stack([preprocess_image(np.array(img)) for img in images])
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

def segment_image(image):
    return slic(image, n_segments=100, compactness=10)

input_dirs = [
    './Celeb-DF/train/Celeb-real',
    './Celeb-DF/train/Celeb-synthesis',
    './Celeb-DF/train/YouTube-real'
]

output_dirs = [
    './Celeb-DF/train/LIME/VGG19/Celeb-real',
    './Celeb-DF/train/LIME/VGG19/Celeb-synthesis',
    './Celeb-DF/train/LIME/VGG19/YouTube-real'
]

for input_dir, output_dir in zip(input_dirs, output_dirs):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
        print(f"Warning: No images found in {input_dir}")
        continue

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(root, fname)
                relative_path = os.path.relpath(image_path, input_dir)  
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '_lime.png')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image = Image.open(image_path).convert("RGB")
                original_image = np.array(image)


                try:
                    explanation = explainer.explain_instance(
                        original_image,
                        model_predict,
                        top_labels=2,
                        hide_color=0,
                        num_samples=100,
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

                    Image.fromarray(combined_image.astype(np.uint8)).save(output_path)

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")  

                # Optionally visualize the output
                # plt.imshow(combined_image)
                # plt.title(f"LIME explanation for {image_name}")
                # plt.axis('off')
                # plt.show()
