# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:03:01 2019

@author: Quang
"""

import numpy as np
import cv2
from keras.preprocessing import image

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
"""
A Haarcascade is bascially a classifier which is ued to detect particular 
object from the source. "Haarcasacade_frontalface_alt is a haarcascade designen
by OpenCV to detect the formal face
"""

cap=cv2.VideoCapture("Kobe.mp4")

from keras.models import model_from_json
# Load facical expression model structure from computer
model = model_from_json(open("facial_expression_model_structure.json","r").read())

# Load facial experession weight from computer
model.load_weights('facial_expression_model_weights.h5')

#emotions=('angry','disgust', 'fear','happy','sad','surpurise','neutral') # Enlgish
emotions=('angry','angry', 'sad','happy','sad','surpurise','neutral') # Enlgish

#emotions=('ärgerlich','ekel', 'angst','glücklich','traurig','überraschung','neutral') # German

_, frame = cap.read()
height,width = frame.shape[:2]


#writer = cv2.VideoWriter("processed.mp4",cv2.VideoWriter_fourcc(*'DIVX'),24,(width,height))

while True:
    try:
        ret , img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        
        #print(faces) # locations of detected faces
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4) # draw rectangle to main image
            
            detected_face=img[int(y):int(y+h),int(x):int(x+w)]#crop detected face
            detected_face =cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) # transform to gray scale
            detected_face=cv2.resize(detected_face,(48,48)) # resize to 48x48
            
            img_pixels = image.img_to_array(detected_face)
            img_pixels=np.expand_dims(img_pixels,axis=0)
            
            img_pixels/=255 #pixels are in scale of [0,255]. normalize all pixels in scale [0,1]
            
            predictions =model.predict(img_pixels) #store probabilities of 7 expression
            max_index=np.argmax(predictions[0])
            
            emotion=emotions[max_index]
            
            # write emotion text above rectangle
            cv2.putText(img,emotion,(int(x+(w/2)-60),int(y+h+65)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        #img =cv2.resize(img,(800,600))    
#        writer.write(img)
        cv2.imshow("img",img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    except:
        break
    
#Release handle to the webcam
cap.release()
#writer.release()
cv2.destroyAllWindows()
