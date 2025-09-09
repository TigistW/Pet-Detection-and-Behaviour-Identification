import cv2
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.alert import send_alert
from behaviour.behavior_rules import BehaviorRules
from detection.pet_detector import PetDetector
from utils.visualization import draw_detections

# Initialize detector, behavior rules, and alert cooldown
detector = PetDetector(model_path="models/yolov8n.pt")
behavior = BehaviorRules()
ALERT_COOLDOWN = 5
last_alert_time = 0

# Webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Starting real-time pet detection with alerts. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Resize for faster detection
    frame_resized = cv2.resize(frame, (640, 640))

    # Run detection
    detections = detector.detect(frame_resized)
    enriched = behavior.update(detections, 0)

    # Check for alert-worthy behaviors
    current_time = time.time()
    for pet in enriched:
        if pet.get("behavior") in ["active near food", "sleeping on couch"]:
            # Only send alert if cooldown passed
            if current_time - last_alert_time > ALERT_COOLDOWN:
                send_alert(
                    pet_label=pet["label"],
                    behavior=pet["behavior"],
                    bbox=pet["bbox"]
                )
                last_alert_time = current_time

    # Draw bounding boxes and labels
    out_frame = draw_detections(frame_resized.copy(), enriched)

    # Display the annotated frame
    cv2.imshow("Real-Time Pet Detection", out_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Real-time detection stopped.")
