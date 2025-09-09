
import streamlit as st
import cv2
import tempfile
import time
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.pet_detector import PetDetector
from src.behaviour.behavior_rules import BehaviorRules
from src.utils.visualization import draw_detections
from src.alerts.alert import send_alert

# ------------------------------
# Initialize detector and behavior
# ------------------------------
detector = PetDetector(model_path="models/yolov8n.pt", device="cpu", conf=0.25)
behavior = BehaviorRules(static_frame_threshold=30, movement_threshold=15)
stframe = st.empty()
last_alert_time = 0

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.title("🐾 Real-Time Pet Detection & Behavior Alerts")
mode = st.sidebar.radio("Choose Mode", ["Webcam", "Upload Video"])
alert_behaviors = st.sidebar.multiselect(
    "Select behaviors to trigger alerts",
    ["sleeping", "active", "running", "near_food", "unknown"],
    default=["near_food", "sleeping"]
)
ALERT_COOLDOWN = st.sidebar.slider("Alert cooldown (seconds)", 5, 60, 10)
frame_skip = st.sidebar.slider("Frame skip for faster processing", 1, 5, 2)

# ------------------------------
# Helper function
# ------------------------------
# global dictionary to store last known bbox per track_id
last_bbox = {}

def process_frame(frame, frame_idx):
    global last_alert_time, last_bbox

    frame_resized = cv2.resize(frame, (460, 460))

    # Run detection every frame_skip frames
    if frame_idx % frame_skip == 0:
        detections = detector.detect(frame_resized)
        enriched = behavior.update(detections, frame_idx)

        # Update last known bbox
        for det in enriched:
            track_id = det.get("track_id")
            if track_id is not None:
                last_bbox[track_id] = det

    else:
        # Use last known bboxes for skipped frames
        enriched = list(last_bbox.values())

    # Draw detections
    out_frame = draw_detections(frame_resized.copy(), enriched)

    # Alerts
    current_time = time.time()
    for pet in enriched:
        if pet["label"] != "Not Pet" and pet.get("behavior") in alert_behaviors:
            if current_time - last_alert_time > ALERT_COOLDOWN:
                send_alert(pet["label"], pet["behavior"], pet["bbox"])
                last_alert_time = current_time

    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
    return out_frame

# def process_frame(frame, frame_idx):
#     """Run detection, behavior, draw bounding boxes, and handle alerts."""
#     global last_alert_time

#     # Resize for faster inference
#     frame_resized = cv2.resize(frame, (460, 460))

#     # Run detection every `frame_skip` frames
#     if frame_idx % frame_skip == 0:
#         detections = detector.detect(frame_resized)
#         enriched = behavior.update(detections, frame_idx)
#     else:
#         enriched = []

#     # Draw detections
#     out_frame = draw_detections(frame_resized.copy(), enriched)

#     # Trigger alerts
#     current_time = time.time()
#     for pet in enriched:
#         if pet["label"] != "Not Pet" and pet.get("behavior") in alert_behaviors:
#             if current_time - last_alert_time > ALERT_COOLDOWN:
#                 send_alert(pet["label"], pet["behavior"], pet["bbox"])
#                 last_alert_time = current_time

#     out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
#     return out_frame

# ------------------------------
# Webcam Mode
# ------------------------------
if mode == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        st.info("Press 'Stop' in the sidebar to exit webcam stream.")
        frame_idx = 0
        stop_button = st.sidebar.button("Stop Webcam")

        while cap.isOpened():
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            out_frame = process_frame(frame, frame_idx)
            stframe.image(out_frame, channels="RGB")
            frame_idx += 1

        cap.release()

# ------------------------------
# Upload Video Mode
# ------------------------------
elif mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        st.info("Processing uploaded video...")
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1 / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out_frame = process_frame(frame, frame_idx)
            stframe.image(out_frame, channels="RGB")
            frame_idx += 1
            time.sleep(delay)  # maintain video speed

        cap.release()
