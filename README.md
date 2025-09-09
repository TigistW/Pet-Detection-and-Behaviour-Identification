
# 🐾 Real-Time Pet Detection & Behavior Alert

A Streamlit app that detects pets in webcam feeds or uploaded videos, tracks their behavior, and triggers alerts for specific behaviors.

---

## Features

* Detect cats, dogs, rabbits, hamsters, and more.
* Track behavior: sleeping, active, running, near food, or unknown.
* Real-time alerts when selected behaviors occur.
* Works with **webcam** or **uploaded videos**.

---

## Installation

1. Clone the repository:

```bash
git clone repo
cd folder
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Download YOLOv8 model weights (example):

```bash
mkdir models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run deployment/app.py
```

1. Open the URL provided (e.g., `http://localhost:8501`).
2. Choose **Webcam** or **Upload Video** mode.
3. Configure alert behaviors and cooldown in the sidebar.
4. View detections and alerts in real-time.

---

## Notes

* Webcam mode requires an accessible camera. If running on a server without a camera, use uploaded video mode.
