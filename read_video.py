import numpy as np
import cv2 as cv
import os
import pygame as pg
import random
from matrix_operations import MatrixOperators
from insert_img import ImgInsertion
from dafoe_quotes import QOUTES

#eventually try audio
#from ffpyplayer.player import MediaPlayer

class VideoReader():
    def __init__(self,w,h):
        self.RESCALED_W, self.RESCALED_H = w,h
        self.CWD = os.getcwd()
        self.QUOTES = QOUTES
        self.QUOTES_iterator = 0

        ###cascades
        self.CASCADE_FLDR = os.path.join(self.CWD,"cascades")
        self.FACE_CASCADE_FILE = os.path.join(self.CASCADE_FLDR,'haarcascade_frontalface_default.xml')
        self.FACE_CASCADE = cv.CascadeClassifier(self.FACE_CASCADE_FILE)
        self.EYE_CASCADE_FILE = os.path.join(self.CASCADE_FLDR,'haarcascade_eye_tree_eyeglasses.xml')
        self.EYE_CASCADE = cv.CascadeClassifier(self.EYE_CASCADE_FILE)
        self.FILTER_FOLDER = os.path.join(self.CWD,'filters')
        self.APPLE_IMG = os.path.join(self.FILTER_FOLDER,'JUICY.png')

        #ensures no jumpy apple (meaning only create img_insert once per face recog)
        self.APPLE_FIRST = True
        self.APPLE_INSERT = None

        ###cv visuals
        self.FONT = cv.FONT_HERSHEY_SCRIPT_COMPLEX
        self.FONT_SCALE = 1
        self.COLOR = (255,255,255)
        self.THICKNESS = 1

        ###pg stuff
        pg.init()
        pg.mixer.init()
        pg.mixer.set_num_channels(8)

        ###music tracking
        self.MUSIC_FLDR = os.path.join(self.CWD,'music')
        self.DEATHNOTE_MUSIC = os.path.join(self.MUSIC_FLDR,'LowOfSolipsism.mp3')
        pg.mixer.music.load(self.DEATHNOTE_MUSIC)
        self.IS_PAUSED = False

        ###matrix stuff
        self.M_OP = MatrixOperators()

        ###face recog params
        self.FACE_DETECTED = False

        self.FACE_X = None
        self.FACE_Y = None
        self.FACE_W = None
        self.FACE_H = None

        ###eye recog coordinates / roi
        self.EYE_DETECTED = False
        self.EYE_RAD = 15
        self.PUPIL_RAD = 10
        self.ROI_GRAY = None
        self.ROI_COLOR = None
        self.E_X = None
        self.E_Y = None
        self.E_W = None
        self.E_H = None

        ###for text motion

        #set bounds of travel (x,y cant move past where the bounds are)
        self.LAST_X = 0
        self.LAST_Y = 0
        self.TEXT_X = None
        self.TEXT_Y = None
        self.TEXT_X_L = None
        self.TEXT_X_R = None
        self.TEXT_Y_L = None
        self.TEXT_Y_R = None

        self.VEL_X = 2
        self.VEL_Y = 1

        self.BUFFER_Y = 10

        self.FRAME_H, self.FRAME_W = None, None

        self.FIRST_FACE = True

    def getSND(self,snd_path):
        snd = pg.mixer.Sound(snd_path)
        return snd

    def read_webcam(self,cap_num):
        cap = cv.VideoCapture(cap_num)
        face_count = [None]*3
        i = 0
        while True: 
            if i>len(face_count)-1:
                i = 0
            #capture frame-by-frame
            ret, frame = cap.read()
            #perform operations

            #flip the frame. I dont like mirror vision!
            frame = cv.flip(frame,1)

            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

            faces = self.FACE_CASCADE.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            #when gets a face: type = np.ndarray
            if type(faces) == np.ndarray:
                #face is detected
                #check if face detected in past 3 frames
                face_count[i] = True
            else:
                face_count[i] = False
            i+=1
            if any(face_count):
                self.FACE_DETECTED = True
                if self.FIRST_FACE:
                    self.FRAME_H, self.FRAME_W = frame.shape[:2]
                    pg.mixer.music.play(-1)
                    self.FIRST_FACE = False
                #play epic music
                else:
                    if self.IS_PAUSED:
                        pg.mixer.music.unpause()
                        self.IS_PAUSED = False

            elif not any(face_count):
                self.FACE_DETECTED = False
                #reset the position of the text
                self.LAST_X = 0
                self.LAST_Y = 0
                if not self.IS_PAUSED:
                    pg.mixer.music.pause()
                    self.IS_PAUSED = True

            if self.FACE_DETECTED:
                if self.APPLE_FIRST:
                    self.APPLE_INSERT = ImgInsertion(self.APPLE_IMG,self.FRAME_W)
                    self.QUOTES_iterator +=1
                    if self.QUOTES_iterator > len(self.QUOTES)-1:
                        self.QUOTES_iterator = 0
                self.APPLE_FIRST = False


                for x,y,w,h in faces:
                    #eyes are inside the face
                    self.FACE_X,self.FACE_Y,self.FACE_W,self.FACE_H = x,y,w,h
                    #frame = cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

                    #eye stuff,roi = region of interest, cool abbreviation!
                    self.ROI_GRAY = gray[self.FACE_Y:self.FACE_Y+self.FACE_H,
                    self.FACE_X:self.FACE_X+self.FACE_W]
                    self.ROI_COLOR = frame[self.FACE_Y:self.FACE_Y+self.FACE_H,
                    self.FACE_X:self.FACE_X+self.FACE_W]

                    eyes = self.EYE_CASCADE.detectMultiScale(self.ROI_GRAY,scaleFactor=1.2, minNeighbors=5)
                    if type(eyes) == np.ndarray:
                        self.EYE_DETECTED = True
                    else:
                        self.EYE_DETECTED = False
                    
                    if self.EYE_DETECTED:
                        for(ex,ey,ew,eh) in eyes:
                            
                            self.E_X = ex
                            self.E_Y = ey
                            self.E_W = ew
                            self.E_H = eh

                            #x + width /2 = center YOU IDIOT IT IS NOT AVERAGE OF BOTH X'S COME ON MAN
                            center = (self.E_X + (self.E_W//2), self.E_Y +(self.E_H)//2)

                            #cv.rectangle(self.ROI_COLOR,(self.E_X,self.E_Y),(self.E_X+self.E_W,self.E_Y+self.E_H),(0,255,255),3)

                            cv.circle(self.ROI_COLOR,center,self.EYE_RAD,(255,255,255),-1)
                            cv.circle(self.ROI_COLOR,center,self.PUPIL_RAD,(0,0,0),-1)

                    
                    frame = self.M_OP.apply_all_effects_face(frame,self.FACE_X,self.FACE_Y,self.FACE_W,self.FACE_H,brightness=1.25,contrast=1.7)
                    #do the text

                    frame = cv.putText(frame, self.QUOTES[self.QUOTES_iterator], (self.LAST_X,self.LAST_Y), self.FONT, self.FONT_SCALE, 
                    self.COLOR, self.THICKNESS)

                    frame = self.APPLE_INSERT.putImgOnBG(frame)
                
                #now we move the text, just need one check (dont check Y, cause thats fine)
                if self.LAST_X ==0:

                    self.TEXT_X_L = self.FACE_X - int(self.FACE_W * 0.75)
                    self.TEXT_X_R = self.FACE_X + int(self.FACE_W * 0.75)

                    self.TEXT_Y_L = self.FACE_Y - self.BUFFER_Y
                    self.TEXT_Y_R = self.FACE_Y + self.BUFFER_Y

                    #initial point is between the two bounds somewhere
                    self.LAST_X = random.randint(self.TEXT_X_L,self.TEXT_X_R)
                    self.LAST_Y = random.randint(self.TEXT_Y_L,self.TEXT_Y_R)

                elif self.LAST_X !=0:

                    if self.LAST_X < self.TEXT_X_L or self.LAST_X > self.TEXT_X_R:
                        self.VEL_X *=-1
                    
                    if self.LAST_Y < self.TEXT_Y_L or self.LAST_Y > self.TEXT_Y_R:
                        self.VEL_Y *=-1

                    self.LAST_X += self.VEL_X
                    self.LAST_Y += self.VEL_Y



                #show the image, apply filters

                #print("X:{}:{}".format(self.FACE_X,self.FACE_W))
                #print("Y:{}:{}".format(self.FACE_Y,self.FACE_H))
            
            elif not self.FACE_DETECTED:
                self.APPLE_FIRST = True


                
            
            #just resize the frame
            frame = cv.resize(frame,(self.RESCALED_W,self.RESCALED_H))
            cv.imshow('Shinigami Eyes',frame)
            

            if cv.waitKey(1) & 0xFF == ord('q'):
                break


        #always release the capture
        cap.release()
        cv.destroyAllWindows()


        
    


