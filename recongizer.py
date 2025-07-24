import cv2
import os
import numpy as np

class Recognizer:
    def __init__(self, trainer_dir='trainer', cascade_dir='haarcascades'):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Ruta absoluta al archivo trainer.yml
        self.trainer_dir = os.path.join(base_dir, '..', trainer_dir)
        trainer_path = os.path.join(self.trainer_dir, 'trainer.yml')

        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"No se encontró el archivo de entrenamiento en {trainer_path}")

        # Carga el reconocedor LBPH y el modelo entrenado
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(trainer_path)

        # Ruta absoluta al Haar Cascade
        cascade_path = os.path.join(base_dir, cascade_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"No se pudo cargar Haar Cascade desde {cascade_path}")

    def recognize(self, img):
        # Convierte a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecta rostros
        faces = self.face_cascade.detectMultiScale(gray)

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            # Predice id y confianza
            id_, confidence = self.recognizer.predict(face)

            results.append({
                "id": int(id_),
                "confidence": float(confidence),
                "rect": (int(x), int(y), int(w), int(h)),
            })
        return results
