import cv2

#face Classifier
face_cascade = cv2.CascadeClassifier ('./Classifiers/haarcascade_frontalface_default.xml')

# eyes classifier
eye_cascade = cv2.CascadeClassifier('./Classifiers/haarcascade_eye.xml')

# Capture videos
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

# loop 
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
         # To draw rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
    
    cv2.imshow('img',img)

#for quiting the programs by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
