import cv2

def draw_detections(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det['bbox']
        label = det.get('label','pet')
        behavior = det.get('behavior')
        conf = det.get('conf',0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"{label} {conf:.2f}"
        if behavior:
            text += f" | {behavior}"
        cv2.putText(frame, text, (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame