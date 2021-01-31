import numpy as np
import pandas as pd
import cv2 as cv
import os
from insert_img import ImgInsertion

class MatrixOperators():
    #The frame of the capture is a np.array
    def __init__(self):
        self.CWD = os.getcwd()
        self.FILTER_FLDR = os.path.join(self.CWD,'filters')
        self.FILTER_SPEED = os.path.join(self.FILTER_FLDR,'SPEED.png')
        self.FILTER_SPEED_IMG = cv.imread(self.FILTER_SPEED)
        self.EYE_RIGHT_FILTER = os.path.join('eye_right.png')
        self.EYE_RIGHT_IMG = cv.imread(self.EYE_RIGHT_FILTER)

    def getRedArray(self,frame):
        #get dimensions as a tuple (m x n x color channels)
        #personal webcam is (480, 640)
        DIMENSIONS = frame.shape[:2]
        BLANK = np.zeros(DIMENSIONS,dtype='uint8')
        
        R = cv.split(frame)[-1]
        #merge B,G,R
        RED = cv.merge([BLANK,BLANK,R])
        return RED
    
    def apply_emphasis_lines(self,frame):
        filtered = cv.addWeighted(frame,0.7,self.FILTER_SPEED_IMG,0.3,0)
        return filtered
    
    def zoom_in_face(self,frame,x,y,w,h):
        #TO DO: FIX THIS UP PLEASE
        frame_h,frame_w = frame.shape[:2]
        BUFFER_X,BUFFER_Y = int(frame_w*.2),int(frame_h*.2)
        #print(frame_h,frame_w)
        minX,maxX = x-BUFFER_X,x+w+BUFFER_X
        minY,maxY = y-BUFFER_Y,y+h+BUFFER_Y

        #print("X:{}:{}".format(minX,maxX))
        #print("Y:{}:{}".format(minY,maxY))

        cropped = frame[minX:maxX,minY:maxY]
        try:
            resized = cv.resize(cropped,(frame_w,frame_h))
            return resized
        except:
            return frame
    
    def change_saturation(self,frame,brightness,contrast):

        frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        
        #clip so dont go above 255
        frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness,0,255)

        frame = cv.cvtColor(frame,cv.COLOR_HSV2BGR)

        return frame

        

    def apply_all_effects_face(self,frame,x,y,w,h,brightness,contrast):
        #applies all effects
        frame = self.getRedArray(frame)
        frame = self.zoom_in_face(frame,x,y,w,h)
        frame = self.apply_emphasis_lines(frame)
        frame = self.change_saturation(frame,brightness,contrast)


        return frame




        
    






        

