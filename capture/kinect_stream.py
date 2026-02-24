from pykinect2 import PyKinectV2, PyKinectRuntime
from pykinect2.PyKinectV2 import *
import numpy as np
import cv2

class KinectCapture:
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Infrared
        )

    def get_color_frame(self):
        if self.kinect.has_new_color_frame():
            frame = self.kinect.get_last_color_frame()
            return frame.reshape((1080, 1920, 4)).astype(np.uint8)
        return None

    def get_depth_frame(self):
        if self.kinect.has_new_depth_frame():
            frame = self.kinect.get_last_depth_frame()
            return frame.reshape((424, 512)).astype(np.float32)
        return None

    def get_ir_frame(self):
        if self.kinect.has_new_infrared_frame():
            frame = self.kinect.get_last_infrared_frame()
            return frame.reshape((424, 512)).astype(np.float32)
        return None

    def close(self):
        self.kinect.close()