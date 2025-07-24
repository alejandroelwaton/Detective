import cv2
import os
import numpy as np

class Recognizer:
    def __init__(self, trainer_dir='trainer', cascade_dir='haarcascades'):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.trainer_dir = os.path.join(base_dir, '..', trainer_dir)
        trainer_path = os.path.join(self.trainer_dir, 'trainer.yml')

        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"Trainer path cannot be found (path)-> {trainer_path}")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(trainer_path)

        cascade_path = os.path.join(base_dir, cascade_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"HaarCascade path ()-> {cascade_path} cannot be found")

    def recognize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray)

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            id_, confidence = self.recognizer.predict(face)

            results.append({
                "id": int(id_),
                "confidence": float(confidence),
                "rect": (int(x), int(y), int(w), int(h)),
            })
        return results
