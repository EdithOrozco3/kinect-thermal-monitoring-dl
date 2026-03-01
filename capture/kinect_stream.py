import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykinect2 import PyKinectV2, PyKinectRuntime
import numpy as np
from config import (KINECT_COLOR_WIDTH, KINECT_COLOR_HEIGHT,
                    KINECT_IR_WIDTH, KINECT_IR_HEIGHT)

class KinectCapture:
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Infrared
        )
        print('[KinectCapture] Sensor inicializado.')

    def get_color_frame(self):
        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame()
            return frame.reshape((KINECT_COLOR_HEIGHT,
                                   KINECT_COLOR_WIDTH, 4)).astype(np.uint8)
        return None

    def get_depth_frame(self):
        if self._kinect.has_new_depth_frame():
            frame = self._kinect.get_last_depth_frame()
            return frame.reshape((KINECT_IR_HEIGHT,
                                   KINECT_IR_WIDTH)).astype(np.float32)
        return None

    def get_ir_frame(self):
        if self._kinect.has_new_infrared_frame():
            frame = self._kinect.get_last_infrared_frame()
            return frame.reshape((KINECT_IR_HEIGHT,
                                   KINECT_IR_WIDTH)).astype(np.float32)
        return None

    def close(self):
        self._kinect.close()
        print('[KinectCapture] Sensor liberado.')