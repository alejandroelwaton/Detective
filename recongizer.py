import cv2
import os
import numpy as np

class Recognizer:
    def __init__(self, trainer_dir='trainer', cascade_dir='haarcascades'):
        # Directorio base donde está este archivo recognize.py
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Ruta absoluta a la carpeta trainer dentro de detective/
        trainer_path = os.path.abspath(os.path.join(base_dir, trainer_dir))
        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"Trainer path not found: {trainer_path}")

        # Ruta absoluta al archivo trainer.yml
        trainer_file = os.path.join(trainer_path, "trainer.yml")
        if not os.path.isfile(trainer_file):
            print(f"Trainer file not found: {trainer_file}")

        # Crear recognizer y cargar modelo entrenado
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.isfile(trainer_file) or os.path.getsize(trainer_file) == 0:
            print(f"[⚠️] El archivo {trainer_file} no existe o está vacío. El reconocedor no se cargará.")
            self.recognizer = None
        else:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(trainer_file)

        # Ruta absoluta al Haarcascade dentro de detective/
        cascade_path = os.path.abspath(os.path.join(base_dir, cascade_dir, 'haarcascade_frontalface_default.xml'))
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"HaarCascade not found or failed to load: {cascade_path}")

    def recognize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        results = []
        height, width = gray.shape

        for (x, y, w, h) in faces:
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(10, min(w, width - x))
            h = max(10, min(h, height - y))

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            id_, confidence = self.recognizer.predict(face)

            if confidence < 55:
                print(f"Recognized ID {id_} | Confidence: {confidence:.2f}")
                user_id = int(id_)
            else:
                print(f"[⚠️] Unknown | Confidence: {confidence:.2f}")
                user_id = "Unknown"

            results.append({
                "id": user_id,
                "confidence": float(confidence),
                "rect": [int(x), int(y), int(w), int(h)]
            })

        return results
