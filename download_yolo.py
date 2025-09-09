import os
import requests

# Specify model name (YOLOv8 variants: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
MODEL_NAME = "yolov8n.pt"

# Create models folder if it doesn't exist
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

model_path = os.path.join(SAVE_DIR, MODEL_NAME)

if os.path.exists(model_path):
    print(f"[INFO] Model already exists at {model_path}")
else:
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{MODEL_NAME}"
    print(f"[INFO] Downloading {MODEL_NAME} from {url} ...")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Model saved to {model_path}")
    else:
        print(f"[ERROR] Failed to download {MODEL_NAME}, status code: {response.status_code}")
