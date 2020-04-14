# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:10:29 2020

@author: Quang
"""

import numpy as np
import cv2

def anonymize_face_simple(image, factor=3.0):
    # automatically detect the size of the blurring kernel based 
    # on the spatial demenison of the input image
    
    (h,w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    
    # essure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
        
    # Apply a Gaussian blur to the input image using our computed
    # kernel size
    
    return cv2.GaussianBlur(image,(kW,kH),0)

def anonymize_face_pixelate (origin_image,blocks=3):
    # divide the input image into NxN blocks
    image = origin_image.copy()
    (h,w) = image.shape[:2]
    xSteps = np.linspace(0,w,blocks + 1, dtype="int")
    ySteps = np.linspace(0,h,blocks + 1, dtype="int")
    
    # Loop over the blocks in both the x and the y direction
    
    for i in range(1,len(ySteps)):
        for j in range(1,len(xSteps)):
            # compute the starting and ending (x,y)-coordinates 
            # for the current block
            startX = xSteps[j-1]
            startY = ySteps[i-1]
            endX = xSteps[j]
            endY = ySteps[i]
            
            # extract the ROI using Numpy array slicing, compute 
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B,G,R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image,(startX,startY),(endX,endY),
            (B,G,R), -1)
    # return the pixelated blurred image
    return image
    
if __name__=="__main__":
    image = cv2.imread("chris_evans.png")
    test_image_gaussian = anonymize_face_pixelate(image)
    test_image_blocks = anonymize_face_simple(image)
    stack_image = np.concatenate((image,test_image_blocks,test_image_gaussian),axis=1)
    cv2.imshow("Blur with gaussian/block",stack_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            