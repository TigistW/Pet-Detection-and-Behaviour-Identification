"""Entry point: run webcam -> detection -> rule-based behavior -> console alert
"""
import time
import argparse
from src.detection.pet_detector import PetDetector
from src.behaviour.behavior_rules import BehaviorRules
from src.utils.video_stream import VideoStream
from src.utils.visualization import draw_detections
from src.alerts.console_alert import alert_console
import cv2

def main(video_src=0, model_path='yolov8n.pt'):
    detector = PetDetector(model_path=model_path, device='cpu', conf=0.25)
    # example ROI for food bowl; adapt coordinates after you know camera
    roi = None # e.g., (400, 300, 600, 480)
    behavior = BehaviorRules(static_frame_threshold=30, movement_threshold=10, roi=roi)
    stream = VideoStream(video_src)
    frame_idx = 0
    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                print('No frame — exiting')
                break
            detections = detector.detect(frame)
            enriched = behavior.update(detections, frame_idx)

            # raise alert for particular behaviors
            for det in enriched:
                if det['behavior'] == 'near_food':
                    alert_console(f"{det['label']} near food at frame {frame_idx}")
                if det['behavior'] == 'sleeping':
                    # example: only alert if sleeping detected for certain frames
                    alert_console(f"{det['label']} detected sleeping (frame {frame_idx})")

            out = draw_detections(frame, enriched)
            cv2.imshow('Pet Monitor', out)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=int, default=0)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    args = parser.parse_args()
    main(video_src=args.src, model_path=args.model)