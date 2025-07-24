import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

base_dir = os.path.dirname(os.path.abspath(__file__))

cascade_path = os.path.join(base_dir, 'haarcascades', 'haarcascade_frontalface_default.xml')

print(f"[INFO] Using casacade at: {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)
def getImagesAndLabels(dataset_path):
    face_samples = []
    ids = []

    for user_id in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_id)
        if not os.path.isdir(user_folder):
            continue

        for filename in os.listdir(user_folder):
            img_path = os.path.join(user_folder, filename)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            gray_img = Image.open(img_path).convert('L')
            img_numpy = np.array(gray_img, 'uint8')

            faces = face_cascade.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(int(user_id))

    return face_samples, ids

def train_model():
    print("[INFO] Getting train model done")

    dataset_path = "dataset"
    faces, ids = getImagesAndLabels(dataset_path)

    if len(faces) == 0:
        return f"[WARN] no faces at {dataset_path}"

    recognizer.train(faces, np.array(ids))

    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')

    print(f"[INFO] Training done with -> {len(np.unique(ids))} users.")
    return f"Entrenamiento completado con {len(np.unique(ids))} usuarios."
