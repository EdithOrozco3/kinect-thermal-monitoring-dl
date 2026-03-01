import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import cv2
from config import MODEL_PATH, MODEL_INPUT_SIZE, FEVER_THRESHOLD

class FeverPredictor:
    def __init__(self):
        self.model     = tf.keras.models.load_model(MODEL_PATH)
        self.threshold = FEVER_THRESHOLD
        print(f'[Predictor] Modelo cargado.')

    def _preprocess_rgb(self, roi_bgra):
        img = cv2.cvtColor(roi_bgra, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, MODEL_INPUT_SIZE)
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    def _preprocess_ir(self, roi_ir):
        img = cv2.resize(roi_ir, MODEL_INPUT_SIZE)
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=-1)
        return np.expand_dims(img, axis=0)

    def predict(self, rgb_roi, ir_roi):
        prob = float(self.model.predict(
            [self._preprocess_rgb(rgb_roi),
             self._preprocess_ir(ir_roi)], verbose=0)[0][0])
        return prob >= self.threshold, prob