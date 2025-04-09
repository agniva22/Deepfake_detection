import os
import shap
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def replace_inplace_relu(model):
    for module_name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        else:
            replace_inplace_relu(module)
    return model

model = models.alexnet(pretrained=True)
model.eval()

model = replace_inplace_relu(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),          
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = './FF++/test/original_sequences_actors'
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0)  
explainer = shap.DeepExplainer(model, input_image)
shap_values = explainer.shap_values(input_image)
output_dir = './FF++/test/SHAP/Alexnet/original_sequences_actors'
os.makedirs(output_dir, exist_ok=True)

shap.summary_plot(shap_values[0], input_image.squeeze().cpu().numpy(), plot_type="bar", show=False)
summary_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
plt.savefig(summary_plot_path)

shap_image_path = os.path.join(output_dir, 'shap_image.png')
shap.image_plot(shap_values, input_image.squeeze().cpu().numpy(), show=False)
plt.savefig(shap_image_path)

print(f"SHAP plots saved to {output_dir}")
