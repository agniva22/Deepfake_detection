import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import shap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='shap_extraction_resnet50.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load ResNet50 model
model = ResNet50(weights='imagenet')

def load_image(image_path, img_size=(224, 224)):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        
        img = cv2.resize(img, img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

# Choose your folder
folder_to_use = './FF++/test/original_sequences_actors'
# folder_to_use = './FF++/test/original_sequences_youtube'
# folder_to_use = './FF++/test/manipulated_sequences'

image_names = os.listdir(folder_to_use)

def model_predict(x):
    x = preprocess_input(x)
    return model.predict(x)

# Use one sample image to initialize the masker
sample_image = load_image(os.path.join(folder_to_use, image_names[0]))
masker = shap.maskers.Image("inpaint_telea", sample_image[0].shape)
explainer = shap.Explainer(model_predict, masker)

# Output directory for SHAP explanations
shap_output_dir = './FF++/test/SHAP/ResNet50/original_sequences_actors'
os.makedirs(shap_output_dir, exist_ok=True)

for image_name in image_names:
    if not image_name.lower().endswith(('jpg', 'jpeg', 'png')):
        continue

    image_path = os.path.join(folder_to_use, image_name)
    
    try:
        selected_image = load_image(image_path)
        if selected_image is None:
            continue

        shap_values = explainer(selected_image, max_evals=500, batch_size=50)

        shap.image_plot(shap_values, selected_image, show=False)

        print(f"Number of axes in the plot: {len(plt.gcf().axes)}")

        shap_image_path_full = os.path.join(shap_output_dir, f"{os.path.splitext(image_name)[0]}_shap_full.png")
        plt.gcf().savefig(shap_image_path_full)

        if len(plt.gcf().axes) > 1:
            output_0_ax = plt.gcf().axes[1]
        else:
            output_0_ax = plt.gcf().axes[0]

        output_0_ax.set_xticks([])
        output_0_ax.set_yticks([])
        output_0_ax.set_xlabel('')
        output_0_ax.set_ylabel('')
        output_0_ax.set_title('')

        extent = output_0_ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        shap_image_path = os.path.join(shap_output_dir, f"{os.path.splitext(image_name)[0]}_shap.png")
        plt.gcf().savefig(shap_image_path, bbox_inches=extent, pad_inches=0, transparent=True)

        plt.close()

        logging.info(f"Saved SHAP output 0 plot without labels for: {image_name}")

    except Exception as e:
        logging.error(f"Error processing image {image_name}: {e}")
