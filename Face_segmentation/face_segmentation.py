# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:01:44 2020

@author: Quang
"""
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-v", "--video", type=str, 
                default=os.path.join("examples","test_video.mp4"),
                help="path to input video")
ap.add_argument("-f","--face", default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-c", "--iter", type=int, default=10,
                help="# of GrabCut iterations (larger value => slower runtime)")
ap.add_argument("-conf", "--confidence", type=float, default=0.5,
                help="face detector filter threshold")                
ap.add_argument("-of","--offset", type=int, default=25,
                help="offset of boudning box around detected faces")
ap.add_argument("-o", "--output", default="output",
                help="path of processed ouput video")
ap.add_argument("-s", "--show", default="y",
                help="show video or not (y/n")
args = vars(ap.parse_args())


values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)
offset=args["offset"]

def load_face_detector(face_detector_path):
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.join(face_detector_path, "deploy.prototxt")
    weightsPath = os.path.join(face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] model loaded successfully...")
    return faceNet

def extract_face_rect(image, faceNet):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    rects=[]
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > args["confidence"]:
            # compute the coordinates of bounding box on detected face
            box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
            
            # append those coorinates in a list 
            (startX, startY, endX, endY) = box.astype("int")
            rect = (startX-offset, startY-offset,endX+offset, endY+offset)
            rects.append(rect)
    return rects
            
def HSV_filter(image):        
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([179, 255, 169])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def process_image(image, faceNet, values):           
    # generate black output mask 
    #mask = np.zeros(shape=image.shape[:2], dtype=np.int8)
    mask = np.ones(shape=image.shape[:2], dtype=np.int8)
    # extract face coordinations 
    rects = extract_face_rect(image, faceNet)
    
    if rects:    
        # allocate memory for two arrays that the Grabcuts algorithm internally
        # uses when segmenting the foregound from background
        fgModel = np.zeros(shape=(1,65), dtype="float")
        bgModel = np.zeros(shape=(1,65), dtype="float")
        
#        start = time.time()
        # apply Grabcut using the bounding box segmentation method 
        (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rects[0], bgModel, 
        fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)
#        end = time.time()
#        print("[INFO] Segmenting process took {:.2f} seconds".format(end-start))
        
        # set all definete background and probable background pixels 
        # to 0 while definite foreground and probable foreground pixels
        # are set to 1
        outputMask = np.where((mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD), 0, 1)    
        
        # scale the mask from the range [0,1]  to [0,255]
        outputMask = (outputMask * 255).astype("uint8")
        
        # apply a bitwise AND to the image using our mask generated 
        # by GrabCut to generate final output image
        output = cv2.bitwise_and(image, image, mask=outputMask)
        output = np.where(output==0, output+255, output)
        
        
        return outputMask, output
    
def process_video(video_path, faceNet, values):

    cap = cv2.VideoCapture(video_path)
    
    # configuation ouput video
    number_frame = 18.0 
    video_size = (1280*2,720)
    writer = cv2.VideoWriter("{}.avi".format(args["output"]),cv2.VideoWriter_fourcc(*'XVID'),number_frame,video_size)
    
    while True:
        try:
            ret, frame = cap.read()
            
            if frame is None:
                break
                
            frame_hsv = HSV_filter(frame)
            ouputMask, output = process_image(frame_hsv, faceNet, values)
            
            #cv2.imshow("Input", frame)
            #cv2.imshow("GrabCut Mask", outputMask)
            #cv2.imshow("GrabCut Output", output)
            
            stack_image = np.hstack((frame, output))

            if args["show"]=="y":
                cv2.imshow("Result", stack_image)
                
            writer.write(stack_image)
            key = cv2.waitKey(1) & 0xff
            
            if key== 27:
                break
        except:
            pass
        
    cap.release()
    cv2.destroyAllWindows()
        
    
if __name__=="__main__":
    
    faceNet=load_face_detector(args["face"])
    
    if args["image"]!=None:
        image = cv2.imread(args["image"])
        outputMask, output = process_image(image, faceNet, values)
        cv2.imshow("Input", image)
        cv2.imshow("GrabCut Mask", outputMask)
        cv2.imshow("GrabCut Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        process_video(args["video" ], faceNet, values)
