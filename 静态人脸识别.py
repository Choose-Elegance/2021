import cv2 as cv
img = cv.imread(r'C:\Users\Lenovo\Pictures\Saved Pictures\photo2.jpg')
def fac_detect_demo():
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(r'E:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
    faces = face_detector.detectMultiScale(gray_img)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result_img',img)
fac_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()

