from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
from src.detection.pet_detector import PetDetector
from src.behaviour.behavior_rules import BehaviorRules
from src.utils.visualization import draw_detections
import base64
from io import BytesIO
import time

app = FastAPI()

# Initialize detector & behavior rules
detector = PetDetector(model_path='models/yolov8n.pt')
behavior = BehaviorRules()

@app.post('/detect/image')
async def detect_image(file: UploadFile = File(...)):
    start_time = time.time()
    print("[STEP 1] Received request")

    # read bytes
    contents = await file.read()
    print(f"[STEP 2] Read {len(contents)} bytes from uploaded file")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("[ERROR] Failed to decode image")
        return {"error": "Invalid image file"}
    print("[STEP 3] Image decoded successfully")

    # Resize
    img_resized = cv2.resize(img, (640, 640))
    print("[STEP 4] Image resized to 640x640")

    # Detection
    detections = detector.detect(img_resized)
    print(f"[STEP 5] Detection completed: {len(detections)} objects found")

    # Behavior analysis
    enriched = behavior.update(detections, 0)
    print("[STEP 6] Behavior analysis completed")

    # Draw results
    out_img = draw_detections(img_resized.copy(), enriched)
    _, buffer = cv2.imencode(".jpg", out_img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    print("[STEP 7] Image annotated and encoded to base64")

    end_time = time.time()
    print(f"[INFO] Total processing time: {end_time - start_time:.2f} seconds")

    return {"detections": enriched, "image": jpg_as_text}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
