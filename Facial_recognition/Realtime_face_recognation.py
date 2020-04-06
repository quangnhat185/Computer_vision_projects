import face_recognition
import cv2


# Get a reference to webcam #0
cap =cv2.VideoCapture("Taylor.mp4")

# Load a sample picture and learn how to recognize it

taylor_image=face_recognition.load_image_file("Taylor.jpg")
taylor_face_encoding=face_recognition.face_encodings(taylor_image)[0]

ed_image=face_recognition.load_image_file("Ed.jpg")
ed_face_encoding=face_recognition.face_encodings(ed_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings =[taylor_face_encoding, ed_face_encoding]
known_face_names=["Taylor Swift", "Ed Sheeran"]


# Initialize some variables
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

_, frame = cap.read()
height,width = frame.shape[:2]

writer = cv2.VideoWriter("processed.mp4",cv2.VideoWriter_fourcc(*'DIVX'),24,(width,height))

while True:
    try:
        # Grab a single frame of video
        ret,frame=cap.read()
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        resize_value=0.25
        small_frame=cv2.resize(frame,(0,0),fx=resize_value,fy=resize_value)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame=small_frame[:,:,::-1]
        
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
            
            face_names=[]
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches =face_recognition.compare_faces(known_face_encodings,face_encoding)
                name="Unknow"
                
                # If a match was found in known_face_encodings, just use the first one
                if True in matches:
                    first_match_index=matches.index(True)
                    name=known_face_names[first_match_index]
                    
                face_names.append(name)
        
        process_this_frame= not process_this_frame
        
        ratio=int(1/resize_value)
        # Display the reuslts
        for (top,right,bottom,left), name in zip(face_locations,face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top*=ratio
            right*=ratio
            bottom*=ratio
            left*=ratio
            
            # Draw a box around the face
            cv2.rectangle(frame,(left,top),(right,bottom),(128,0,128),2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame,(left,bottom-35),(right,bottom),(128,0,128),cv2.FILLED)
            cv2.putText(frame,name,(left+6,bottom-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            
        # Display the resulting image
        #frame=cv2.resize(frame,(640,480))
        writer.write(frame)
        cv2.imshow("AI_Facial Recognition",frame)
        
        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
    except:
        break
# Release handle to the webcame
cap.release()
writer.release()
cv2.destroyAllWindows()
    
    
    
    
    