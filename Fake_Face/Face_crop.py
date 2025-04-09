import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_faces_from_image(image_path, output_folder):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for idx, (x, y, w, h) in enumerate(faces):
        cropped_face = img[y:y+h, x:x+w]
        face_filename = os.path.join(output_folder, f"face_{idx+1}_{os.path.basename(image_path)}")
        cv2.imwrite(face_filename, cropped_face)

def process_images_in_folders(input_folders, output_root_folder):
    for folder in input_folders:
        folder_name = os.path.basename(folder)
        output_folder = os.path.join(output_root_folder, folder_name)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            if os.path.isfile(image_path):
                crop_faces_from_image(image_path, output_folder)
                print(f"Processed {filename} in folder {folder_name}")

input_folders = ['easy', 'mid', 'hard', 'real']
output_root_folder = 'cropped_faces'

if not os.path.exists(output_root_folder):
    os.makedirs(output_root_folder)

process_images_in_folders(input_folders, output_root_folder)
