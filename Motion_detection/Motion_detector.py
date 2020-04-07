import datetime
import imutils
import time
import cv2


# directory to input video

cap = cv2.VideoCapture("time_square.mp4")

time.sleep(2.0)

# initialize the first fram in the video stream
firstFrame=None

_, frame = cap.read()
height,width = frame.shape[:2]

def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]

    elif len(cnts) == 3:
        cnts = cnts[1]

    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))
        
    return cnts
writer = cv2.VideoWriter("processed.mp4",cv2.VideoWriter_fourcc(*'DIVX'),24,(int(768*(width/height)),768))

while True:
    
    #grab the current frame and initialize the movement/no_movement text
    _, frame = cap.read()
    text="No Comment"
    
	# if the frame could not be caught, then we have reached the end of the video
    if frame is None:
        break
    
	# resize the frame, convert it to grayscale, and blur it
    frame = cv2.resize(frame, (int(768*(width/height)),768))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15,15), 0)

	# if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
	# compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 40, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    	# loop over the contours
    for c in cnts: 
		# compute the bounding box for the contour, draw it on the frame, and update the text
        (x,y,w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Movement Detected!"
    
	# draw the text on the frame
    cv2.putText(frame, "Status: {}".format(text), (5, 740), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    #show the frame and record if the user presses a key
    writer.write(frame)
    cv2.imshow("Room Camera",frame)
    cv2.imshow("Threshold View",thresh)
    cv2.imshow("Frame Delta View", frameDelta)
    key=cv2.waitKey(10) & 0xFF
    
    #If the "q" key is presses, break from the loop
    if key==ord("q"):
        break
    
#clean up the camera and close any open windown
cap.release()
writer.release()
cv2.destroyAllWindows()