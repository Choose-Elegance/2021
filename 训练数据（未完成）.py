import cv2 as cv
import numpy as np
from PIL import Image
import os
path = 'E:/Python/train_data'
recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier(r'E:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml');
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples =[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePaths)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print('\n [INFO] 正在训练模型...')
faces,ids = getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))
recognizer.write('trainer.yml')
print ('\n [INFO] {0} 训练成功，正在退出程序'.format(len(np.unique(ids))))
