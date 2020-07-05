# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:51:19 2020

@author: Quang
"""


import numpy as np
import configparser
import cv2

def initialize_config():
    paser  = configparser.ConfigParser()
    paser.read("./config.ini") 
    return paser["DEFAULT"]

config = initialize_config()

def detect_people(frame, net, ln, personIdx):
    """
    grab the dimensions of the frame and initialize
    the list of results
    """
    (H,W) = frame.shape[:2]
    results = []
    
    """
    passing frame through yolo object detector as blob
    """
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    # initialize our lists of detected bounding boxes, centroids, confidence
    boxes = []
    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence probability 
            # of the currenct object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter human detection by threshold confidence
            if classID == personIdx and confidence > float(config['MIN_CONF']):
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype('int')
                
                # derive the top and left corner of the bounding box
                x = int(centerX - (width/2))
                y = int(centerY - (width/2))
                
                # update our list boxes, centroids and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
            
    # apply non-maxima suppresion to suppress weak, overalpping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, float(config['MIN_CONF']), float(config['NMS_THRESH']))
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            """
            update our result list to consist of the person
            prediction probability, bounding box coordinates
            and the centroid
            """
            r = (confidences[i], (x, y , x+w, y+h), centroids[i])
            results.append(r)
    
    # return the list of results            
    return results