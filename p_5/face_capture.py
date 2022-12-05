import numpy as np
import cv2
import os
if not os.path.exists('images'):
    os.makedirs('images')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

detect_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n enter user id (MUST be an integer) and press <return> -->  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.putText(img, "Abner Lourdes Noel", (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k < 30:
        break
    elif count >= 30:
         break
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
