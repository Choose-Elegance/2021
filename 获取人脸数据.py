import cv2 as cv
import os
face_detector = cv.CascadeClassifier(r'E:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
face_id = input('\n 请输入您的ID并回车 ==> ')
print('\n [INFO] 准备拍摄，请看摄像头并等待...')
count = 0
while True :
    ret,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        count += 1
        cv.imwrite(r'E:\Python\train_data/User.' + str(face_id) + '.' + str(count) + '.jpg',gray[y:y+h,x:x+w])
        cv.imshow('image',frame)
    if ord('q') == cv.waitKey(10):
        break
    elif count >= 20:
        break
print('\n [INFO] 拍摄完成，清除无用进程 ')
cap.release()
cv.destroyAllWindows()
