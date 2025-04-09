import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

train_dir = 'cropped_faces/train'

categories = ['easy', 'mid', 'hard']

datagen = ImageDataGenerator(
    rotation_range=30,        
    width_shift_range=0.2,    
    height_shift_range=0.2,  
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,   
    fill_mode='nearest'      
)

for category in categories:
    category_path = os.path.join(train_dir, category)
    
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(category_path, image_file)
        
        img = load_img(image_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=category_path, 
                                  save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 5:  
                break