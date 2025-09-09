
"""Rule-based behavior detection with simple motion heuristics and ROI checks."""

import numpy as np


class BehaviorRules:
    def __init__(self,
                 static_frame_threshold=30,
                 movement_threshold=15,
                 roi=None,
                 ttl_frames=100):
        """
        Args:
            static_frame_threshold: how many frames of near-zero movement to consider "sleeping"
            movement_threshold: pixel distance to consider movement
            roi: (x1,y1,x2,y2) region of interest (e.g., food bowl)
            ttl_frames: remove history if pet not seen for this many frames
        """
        self.static_frame_threshold = static_frame_threshold
        self.movement_threshold = movement_threshold
        self.roi = roi
        self.ttl_frames = ttl_frames
        self.history = {}

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _compute_behavior(self, entry, centroid, bbox, frame_idx, label):
        dx = abs(entry['last_centroid'][0] - centroid[0])
        dy = abs(entry['last_centroid'][1] - centroid[1])
        dist = np.hypot(dx, dy)

        # Default
        behavior = "unknown"

        # Movement-based
        if dist < self.movement_threshold:
            entry['static_count'] += 1
        else:
            entry['static_count'] = 0

        if entry['static_count'] >= self.static_frame_threshold:
            behavior = "sleeping"
        elif dist > self.movement_threshold * 3:
            behavior = "running"
        elif dist >= self.movement_threshold:
            behavior = "exploring"
        else:
            behavior = "active"

        # ROI-based override
        if self.roi is not None:
            x1, y1, x2, y2 = bbox
            rx1, ry1, rx2, ry2 = self.roi
            ix1 = max(x1, rx1)
            iy1 = max(y1, ry1)
            ix2 = min(x2, rx2)
            iy2 = min(y2, ry2)
            if ix2 > ix1 and iy2 > iy1:
                behavior = "near_food"

        # Special case: Not Pet
        if label == "Not Pet":
            behavior = "unknown"

        return behavior, dist

    def update(self, detections, frame_idx: int):
        """
        Args:
            detections: list of dicts with bbox, label, and optional 'track_id'
            frame_idx: current frame number
        Returns:
            detections enriched with 'behavior'
        """
        out = []

        # Clean up history (TTL)
        for k, v in list(self.history.items()):
            if frame_idx - v["last_update"] > self.ttl_frames:
                del self.history[k]

        for i, det in enumerate(detections):
            bbox = det['bbox']
            centroid = self._centroid(bbox)

            # Prefer tracker ID if available, else fallback to label+index
            track_id = det.get("track_id", f"{det['label']}_{i}")

            if track_id not in self.history:
                self.history[track_id] = {
                    "last_centroid": centroid,
                    "static_count": 0,
                    "last_update": frame_idx,
                }

            entry = self.history[track_id]

            # Compute behavior
            behavior, dist = self._compute_behavior(entry, centroid, bbox, frame_idx, det['label'])

            # Update history
            entry['last_centroid'] = centroid
            entry['last_update'] = frame_idx

            out.append({**det, "behavior": behavior, "movement_dist": float(dist)})

        return out
