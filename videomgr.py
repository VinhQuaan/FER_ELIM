import sys

import cv2
import pickle
import numpy as np
import time
from random import shuffle
#from scipy.misc import imread, imresize
from imageio import imread
from timeit import default_timer as timer
from multiprocessing import Process, Lock, Queue
import datetime

class VideoMgr():    
    def __init__(self, camIdx, camName):        
        self.camIdx = camIdx
        self.camName = camName
        self.camCtx = None
        self.start = None
        self.end = None
        self.numFrames = 0

    def open(self, config):        
        self.camCtx = cv2.VideoCapture(self.camIdx)
        if not self.camCtx.isOpened():
            print('isOpend Invalid')
            print(self.camIdx)
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
            
        self.camCtx.set(cv2.CAP_PROP_FRAME_WIDTH, int(config['width']))
        self.camCtx.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config['height']))        
        self.camCtx.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(config['invformat'])))       
        self.camCtx.set(cv2.CAP_PROP_FPS, int(config['fps']))
        
    def read(self):
        return self.camCtx.read()               

    def start(self):
        # start the timer
        self.start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self.end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self.numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self.end - self.start).total_seconds()
         
    def fps(self):
        # compute the (approximate) frames per second
        return self.numFrames / self.elapsed()

    def close(self):
        self.camCtx.release()
    
    def reset(self):
        self.close()
        self.open()