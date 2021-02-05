import cv2 as cv
def fac_detect_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(r'E:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result',img)
cap = cv.VideoCapture(r'C:\Users\Lenovo\Pictures\Camera Roll\WIN_20200923_11_35_36_Pro.mp4')
while True:
    flag,frame = cap.read()
    if not flag:
        break
    fac_detect_demo(frame)
    if ord('q') == cv.waitKey(10):
        break
cv.destroyAllWindows()
cap.release()
