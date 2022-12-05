import cv2
import numpy as np
from PIL import Image
import os
path = './images/'
recog = cv2.face.LBPHFaceRecognizer_create()
#Haar cascade file
detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml');

def getImagesAndLabels(path):
    img_paths = [os.path.join(path,f) for f in os.listdir(path)]
    sample_face=[]
    ids = []
    for imagePath in img_paths:
        # convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detect.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            sample_face.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return sample_face,ids
print ("\n[INFO] Training faces...")
faces,ids = getImagesAndLabels(path)
recog.train(faces, np.array(ids)
recog.write('trainer.yml')
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
