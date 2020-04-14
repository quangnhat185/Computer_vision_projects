# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:04:48 2020

@author: Quang
"""

import numpy as np
import cv2
import time
import os
from utils import anonymize_face_simple, anonymize_face_pixelate

# define the confidence of face detectopm as 0.5
CONFIDENCE = 0.5

def resize_with_ratio(image, width):
    h,w = image.shape[:2]
    ratio = w/h
    image_resize = cv2.resize(image,(width,int(width/ratio)))
    return image_resize

# Load the serialized face detector model from disk
print("[INFO] loading the face detector model...")
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNet(prototxtPath,weightsPath)
print("[INFO]Load model successfully...")

cap = cv2.VideoCapture("Taylor_ed.mp4")
time.sleep(2.0)

writer = cv2.VideoWriter("processed_blured_face.mp4",cv2.VideoWriter_fourcc(*'DIVX'),24,(1980,371))

while True: 
    # read and resize frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = resize_with_ratio(frame,width=660)
    frame_gaussian = frame.copy()
    frame_blocks = frame.copy()
    
    # grab the dimensions of the frame and then construct 
    # a blob from it
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
    (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face
    # detections 
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e: probability) associated 
        # with the detection
        
        confidence = detections[0,0,i,2]
        
        # filter out weak detections by ensureing the confidence is 
        # greater than the minimum confidence
        
        if confidence > CONFIDENCE:
            # compute the (x,y) coordinates of the bounding box for 
            # the objects
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # extract the face ROI
            face = frame[startY:endY,startX:endX] 
            
            # applying "simple" face bluring method
            face_blur_gaussian = anonymize_face_simple(face,factor=3.0)
            
            # applying "pixelated" face anonymization method
            face_blur_blocks = anonymize_face_pixelate(face,blocks=3)
            
            # replace the blurred face in the output image
            frame_gaussian[startY:endY, startX:endX] = face_blur_gaussian
            cv2.putText(frame_gaussian,
                        ("Face blurring with Gasussian technique"),
                        (20,h-20),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255),2)
                        
            frame_blocks[startY:endY, startX:endX] = face_blur_blocks
            cv2.putText(frame_blocks,
                        ("Face blurring with Pixelated technique"),
                        (20,h-20),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255),2)
            
    # show the result
    horizontal_stacked_frame = np.concatenate((frame, frame_gaussian,frame_blocks),axis=1)        
    writer.write(horizontal_stacked_frame)
    cv2.imshow("Face blur with Gaussian/Blocks", horizontal_stacked_frame)
    
    k = cv2.waitKey(1) & 0xFF
    
    # press Esc to quit
    if k == 27:
        break   
    

cap.release()
writer.release()
cv2.destroyAllWindows()