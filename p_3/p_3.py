import numpy as np
import cv2



def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    line = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    line = np.int32(line + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, line, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in line:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr
cap = cv2.VideoCapture('abner.mov')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

#By changing n you can change the amount of frames, hence ran with n=1,11,31
n = 1
count = 0
while suc:

    suc, img = cap.read()
    if(suc and count%n == 0):
        print(count)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
    
    count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
