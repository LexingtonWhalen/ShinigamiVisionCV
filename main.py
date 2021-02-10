###Created by Lex Whalen

import numpy as np
import cv2
import os
from read_video import VideoReader

class MainApp():
    def __init__(self,vid_w,vid_h):
        self.VR = VideoReader(vid_w,vid_h)
        self.CWD = os.getcwd()
    
    def apply_to_webcam(self,cap_num):
        self.VR.read_webcam(cap_num)


if __name__ == '__main__':
    app = MainApp(1080,720)
    app.apply_to_webcam(0)
    
