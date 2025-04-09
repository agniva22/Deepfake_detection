import os
import cv2
import dlib
import glob
import numpy as np
from imutils import face_utils

# Paths
input_dirs = {
    'manipulated_sequences': '/home/ant-pc/papers/FG/FF++/manipulated_sequences/Deepfakes/raw/videos',
    'original_sequences_actors': '/home/ant-pc/papers/FG/FF++/original_sequences/actors/raw/videos',
    'original_sequences_youtube': '/home/ant-pc/papers/FG/FF++/original_sequences/youtube/raw'

}
output_base = "/home/ant-pc/papers/FG/FF+++"
predictor_path = "/home/ant-pc/papers/FG/shape_predictor_81_face_landmarks.dat"

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Function to crop face
def crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    shape_np = face_utils.shape_to_np(shape)
    x, y, w, h = cv2.boundingRect(shape_np)
    face = frame[y:y+h, x:x+w]
    return face

# Process each dataset
for dataset_name, dataset_path in input_dirs.items():
    videos = glob.glob(os.path.join(dataset_path, "*.mp4"))

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 10, dtype=np.int32)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(output_base, dataset_name, video_name)
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue

            face = crop_face(frame)
            if face is not None and face.size != 0:
                save_path = os.path.join(save_dir, f"frame_{count}.jpg")
                cv2.imwrite(save_path, face)
                count += 1
            else:
                print(f"No face detected in {video_name} at frame {idx}. Skipping.")

        cap.release()

        if count == 0:
            print(f"Skipped {video_name} from {dataset_name} (no valid face frames).")
        else:
            print(f"Processed {video_name} from {dataset_name} ({count} frames saved).")


print("All videos processed.")

