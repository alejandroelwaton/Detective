import os
import shutil

BASE_DIR = "dataset"

def save_capture(user_id: str, file_object, filename: str) -> str:
    print(f"Se guardo la captura de {id}")
    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file_object, f)

    return filepath

    
