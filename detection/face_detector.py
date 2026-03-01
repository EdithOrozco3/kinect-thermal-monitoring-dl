import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp
import cv2
from config import FACE_MIN_CONFIDENCE, ROI_PADDING

class FaceDetector:
    def __init__(self):
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            min_detection_confidence=FACE_MIN_CONFIDENCE,
            model_selection=0
        )

    def detect(self, bgra_frame):
        rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
        results = self._detector.process(rgb)
        faces = []
        if results.detections:
            h, w = bgra_frame.shape[:2]
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x  = int(bb.xmin * w)
                y  = int(bb.ymin * h)
                fw = int(bb.width * w)
                fh = int(bb.height * h)
                if fw > 0 and fh > 0:
                    faces.append((x, y, fw, fh))
        return faces

    def extract_roi(self, frame, bbox):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(W, x + w + ROI_PADDING)
        y2 = min(H, y + h + ROI_PADDING)
        roi = frame[y1:y2, x1:x2]
        return None if roi.size == 0 else roi

    def scale_bbox_to_ir(self, bbox):
        x, y, w, h = bbox
        sx, sy = 512/1920, 424/1080
        return (int(x*sx), int(y*sy), int(w*sx), int(h*sy))