###Created by Lex Whalen

import cv2 as cv
import numpy as np
import random

class ImgInsertion():

    def __init__(self,img_path,frame_w):
        self.IMG_PATH = img_path
        self.FRAME_W = frame_w

        self.Y = 0
        self.DY = 10

        ###read image:
        #IMREAD_UNCHANGED keeps the alpha channel
        self.IMG = cv.imread(self.IMG_PATH,cv.IMREAD_UNCHANGED)
        self.H, self.W = self.IMG.shape[:2]

        #randomize x
        self.X = random.randint(0,self.FRAME_W - self.W)

        self.ALPHA_IMG = self.IMG[:,:,3]/255.0
        self.ALPHA_BG = 1.0 - self.ALPHA_IMG

        


    def putImgOnBG(self,bg):
        bg_h,bg_w = bg.shape[:2]
        if self.Y > bg_h-self.H:
            self.Y = 0
            self.X = random.randint(0,self.FRAME_W - self.W)


        for i in range(0,3):
            bg[self.Y:self.Y+self.H,self.X:self.X+self.W,i] = (self.ALPHA_IMG * self.IMG[:,:,i] + 
            self.ALPHA_BG * bg[self.Y:self.Y+self.H,self.X:self.X+self.W,i])

        #move img
        self.Y += self.DY
        return bg




    
        
        
