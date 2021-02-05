import cv2 as cv
import numpy as np
face_detector = cv.CascadeClassifier(r'E:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    ret,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20))
    for x,y,w,h in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
    cv.imshow('result_video',frame)
    if ord('q') == cv.waitKey(10):
        break
cap.release()
cv.destroyAllWindows()
