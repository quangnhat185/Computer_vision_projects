# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:51:40 2020

@author: Quang
"""

# run script
# !python end_to_end_detector.py -v test_video.wmv
# !python end_to_end_detector.py -v test_video.wmv -o output

from utils import initialize_config, detect_people
from scipy.spatial import distance as dist
from bounding_box import bounding_box as bb
import numpy as np
import argparse
import cv2
import os

# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument('-v','--video', type=str, default=False,
                help="path to input video")
ap.add_argument('-o','--output', type=str, default=False,
                help="name of output video")
args = vars(ap.parse_args())

def resize(frame, new_width):
    h, w = frame.shape[:2]
    ratio = w/h
    frame_resize = cv2.resize(frame, (new_width, int(new_width//ratio)))
    
    return frame_resize
    
if __name__=="__main__":
    # initialize config
    config = initialize_config()
    
    colors = {"red": (0, 0 ,255),
              "green": (0, 255, 0)}
    
    #  load yolov3-coco label
    labelsPath = os.path.sep.join([config['model_path'],'coco.names'])
    with open(labelsPath,'r') as file:
        LABELS = file.read().strip().split('\n')
        
    #  Load yolov3-coco object detector 
    weightsPath = os.path.sep.join([config['model_path'],'yolov3.weights'])
    configPath= os.path.sep.join([config['model_path'],'yolov3.cfg'])
    
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLOv3...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    # check if we are going to use GPU
    if bool(int(config["use_gpu"])):
    	# set CUDA as the preferable backend and target
    	print("[INFO] setting preferable backend and target to CUDA...")
    	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # determine only the *output* layer names that wee need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    # intialize the video stream
    
    cap = cv2.VideoCapture(args['video'] if args['video'] else 0)
    
    if (args["output"]):
        fourcc= cv2.VideoWriter_fourcc(*"DVIX")
        writer = cv2.VideoWriter("{:s}.mp4".format(args["output"]), fourcc, 15, (1280, 768), True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = resize(frame, int(config["width_resize"]))
        results = detect_people(frame, net, ln, LABELS.index("person"))
        # initialize the set of indexes that violate the minum social distance
        violate = set()
        
        # ensure there are *at least * two people detections
        
        if len(results) >=2:
            """
            extract all centroid from the results and compute the 
            Euclidean distance between all pairs of the centroids
            """
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            
            # loop over the upper triangular of the distance matrix
            for i in range (0, D.shape[0]):
                for j in range(i+1, D.shape[1]):
                    """
                    check to see if the distacne between any two 
                    centroid pairs is less than the min_distance 
                    pixels in config file
                    """
                    if D[i, j] < int(config['min_distance']):
                        violate.add(i)
                        violate.add(j)
                        
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = "green"
            
            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = "red"
            
            # draw bounding box around the person and his/her centorid
            bb.add(frame,startX,startY-5,endX,endY-5,color=color)
            cv2.circle(frame, (cX, cY), 3, colors[color], -1)
            
        # display the toal number of social distance violation
        text = "Social Distancing Violation: {:d}".format(len(violate))
        cv2.putText(frame, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        
        frame_show = cv2.resize(frame, (1280,768))
        cv2.imshow("Social distancing", frame_show)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        
        """
        Check whether output video file path has been supplied 
        and the video write has not been initialize
        """
        
        if (args["output"]):
            writer.write(frame_show)
            
    cap.release()
    writer.release()
            
            
            
        
        
