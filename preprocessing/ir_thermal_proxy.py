import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from config import (IR_CALIBRATION_FRAMES, IR_GAMMA_CORRECTION,
                    TEMP_NORMAL_MIN, TEMP_FEVER_MIN)

class IRThermalProxy:
    def __init__(self):
        self.baseline   = None
        self._calib_buf = []
        self.calibrated = False

    def add_calibration_frame(self, ir_frame):
        self._calib_buf.append(ir_frame.astype(np.float32))
        if len(self._calib_buf) >= IR_CALIBRATION_FRAMES:
            self.baseline   = np.mean(self._calib_buf, axis=0)
            self.calibrated = True
            print('[IRProxy] Calibración completada.')

    def _correct_by_distance(self, ir_frame, depth_frame):
        d = np.clip(depth_frame / 1000.0, 0.3, 8.0)
        return ir_frame / np.power(d, 3.41)

    def process(self, ir_frame, depth_frame=None):
        frame = ir_frame.astype(np.float32)
        if IR_GAMMA_CORRECTION and depth_frame is not None:
            frame = self._correct_by_distance(frame, depth_frame)
        if self.calibrated:
            frame = np.clip(frame - self.baseline, 0, None)
        p_lo  = np.percentile(frame, 1)
        p_hi  = np.percentile(frame, 99)
        frame = np.clip((frame - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
        return (frame * 255).astype(np.uint8)

    def extract_face_ir(self, ir_frame, depth_frame, ir_bbox):
        processed = self.process(ir_frame, depth_frame)
        x, y, w, h = ir_bbox
        pad = 15
        x1 = max(0, x-pad);    y1 = max(0, y-pad)
        x2 = min(512, x+w+pad); y2 = min(424, y+h+pad)
        roi = processed[y1:y2, x1:x2]
        return None if roi.size == 0 else roi

    def estimate_temperature(self, ir_roi):
        """
        Estima temperatura superficial facial a partir del
        valor IR normalizado de la ROI.
        Mapea el rango [0-255] al rango de temperatura facial
        esperado [35.0°C - 39.5°C] como proxy relativo.
        NOTA: valor estimado, no medición clínica absoluta.
        """
        mean_ir = float(np.mean(ir_roi))
        # Mapeo lineal: IR 0→35.0°C, IR 255→39.5°C
        temp = TEMP_NORMAL_MIN + (mean_ir / 255.0) * (39.5 - TEMP_NORMAL_MIN)
        return round(temp, 1)