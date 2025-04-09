import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
import shap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='shap_extraction.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

model = models.load_model('cnn_model.h5')

def load_image(image_path, img_size=(128, 128)):
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
    
folder_to_use = './FF++/test/original_sequences_actors'

image_names = os.listdir(folder_to_use)

def model_predict(x):
    return model.predict(x)

sample_image = load_image(os.path.join(folder_to_use, image_names[0]))
masker = shap.maskers.Image("inpaint_telea", sample_image[0].shape)
explainer = shap.Explainer(model_predict, masker)

shap_output_dir = './FF++/test/SHAP/CNN/original_sequences_actors'
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

        shap_image_path = os.path.join(shap_output_dir, f"{os.path.splitext(image_name)[0]}_shap.png")

        shap.image_plot(shap_values, selected_image, show=False)

        output_0_ax = plt.gcf().axes[1]

        output_0_ax.set_xticks([])  
        output_0_ax.set_yticks([])
        output_0_ax.set_xlabel('')
        output_0_ax.set_ylabel('')
        output_0_ax.set_title('')

        extent = output_0_ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        plt.gcf().savefig(shap_image_path, bbox_inches=extent, pad_inches=0, transparent=True)

        plt.close()

        logging.info(f"Saved SHAP output 0 plot without labels for: {image_name}")

    except Exception as e:
        logging.error(f"Error processing image {image_name}: {e}")
