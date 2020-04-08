# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:47:28 2020

@author: Quang
"""

from scipy.spatial import distance as dist
import numpy as np
import cv2

def sort_contour(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    
    # handle if we need to sort in reserve     
    if method in ["right-to-left","bottom-to-top"]:
        reverse = True
        
    # handle if we are sorting against the y-coordinate
    # rather than x-coordinate of the bounding box
    if method in ["top-to-bottom","bottom-to-top"]:
        i = 1
        
    # construct the list of bounding boxes and sort them
    # from top to bottom
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                               key=lambda b:b[1][i],reverse=reverse))
    
    return cnts, boundingBoxes


def order_points(pts):
    # sort the ponts based on their x-coordinate
    xSorted = pts[np.argsort(pts[:,0]),:]
    
    # grab the left-most and the right-most points
    # from the sorted coordinate
    leftMost = xSorted[:2,:]
    rightMost = xSorted[2:,:]
    
    # sort the left-most coordinate according to 
    # their y-coordinate so we can grab the top-left
    # and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:,1]),:]
    (tl,bl) = leftMost
    
    # now that we have the top-left coordinate, use it 
    # as an anchor to calculate the Euclidean distance
    # between the top-left and right-most point;
    D = dist.cdist(tl[np.newaxis],rightMost,"euclidean")[0]
    (br,tr) = rightMost[np.argsort(D)[::-1],:]
    
    # return the coordinate in top-left, top-right,
    # bottom-right, and bottom-left order
    
    return np.array([tl,tr,br,bl],dtype="float32")

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] +ptB[1]) * 0.5)

if __name__=="__main__":
         
    # load the image, convert it to grayscale and blur it
#    image = cv2.imread("coins.jpg")
    image = cv2.imread("objects.jpg")
    h,w = image.shape[:2]
    ratio = w/h
    image_resize = cv2.resize(image,(int(480*ratio),480))
    image_forshow = image_resize.copy()
    gray = cv2.cvtColor(image_resize,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(int(480*ratio),480))
#    gray = cv2.GaussianBlur(gray,(15,15),0) # for coins image
    gray = cv2.GaussianBlur(gray,(19,19),0) 
    
    # perform edge detection, then perform a dilation + erosion 
    # to close the gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged_dilate = cv2.dilate(edged, None, iterations=1)
    edged_erode = cv2.erode(edged_dilate, None, iterations=1)
    
    # find contours in the edge map
    cnts = cv2.findContours(edged_erode.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    
    # sort the contours from left-to-right and 
    # initialize the 'pixels per metric' calibration variable
    (cnts, _) = sort_contour(cnts)
    pixelsPerMetric = None
    
    index_label = 1    
    # loop over the contour individually 
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea (c) < 100:
            continue
     
        # compute the rotated bounding box of the contour
        #orig = image_resize.copy()
        orig = image_resize
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box,dtype="int")
        # order the points in the contour such that they appear
        # in top-lef, top-right, bottom-right, and bottom-left
        # oder, then draw the outline of the rotated bounding
        # box
        box = order_points(box)
        cv2.drawContours(orig,[box.astype("int")],-1,(255,0,0),2)
        
        # Loop over the originals points and draw center points
        for (x,y) in box:
            cv2.circle(orig,(int(x),int(y)),5,(0,0,255),-1)
    
        # unpack the ordered bounding box, them compute the midpoint 
        # between the top-left and top-right coordinate, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        
        (tl, tr, br, bl) = box
        (tltrX,tltrY) = midpoint(tl,tr)
        (blbrX,blbrY) = midpoint(bl,br)
        
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint betwen the top-right and bottom-right
        
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # draw the midpoint on the image
        cv2.circle(orig, (int(tltrX),int(tltrY)),5,(255,0,0),-1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    		(0, 255, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    		(0, 255, 255), 2)
        
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX,tltrY), (blbrX,blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        #print (index_label,dA, dB)
        if pixelsPerMetric is None:
            width = float(input("Input the width of the far-left object \n (50cent has the width of 30.61mm): "))
            pixelsPerMetric = dB/width
             
        # compute the size of the object
        if index_label==1:
            dimB = dB / pixelsPerMetric
            dimA = dimB
        else:
            dimA = 0.90 * dA / pixelsPerMetric
            dimB = 0.83 * dB / pixelsPerMetric
        
        print (index_label,dimB)
        
        
        
    	# draw the object sizes on the image
        cv2.putText(orig, "{:.2f}mm".format(dimA),
    		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    		0.5, (0, 0, 255), 2)
        cv2.putText(orig, "{:.2f}mm".format(dimB),
    		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    		0.5, (0, 0, 255), 2)   
        cv2.putText(orig, "{}".format(str(index_label)),
    		(int(tlblX)-20, int(tlblY)-10), cv2.FONT_HERSHEY_SIMPLEX,
    		0.65, (0, 0, 255), 2) 
        
        index_label+=1
    
#    cv2.imwrite("processed_coins.png",np.vstack([image_forshow,orig]))
    cv2.imwrite("processed_objects.png",np.vstack([image_forshow,orig]))
    cv2.imshow("Image",orig)
    key = cv2.waitKey(0) & 0xff
    
    cv2.destroyAllWindows()



img1 = cv2.imread("processed_coins.png")
img2 = cv2.imread("processed_objects.png")

cv2.imwrite("stack_images.png",np.hstack([img1,img2]))


