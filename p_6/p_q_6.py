import cv2
import numpy as np
import matplotlib.pyplot as plt


#Distance from camera to marker = 50cm
#Distance between the 2 positions = 15cm

img_1 = cv2.imread('p1.png', 0) 
img_2= cv2.imread('p2.png', 0)

def ShowDisparity(bSize=5):
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize) 
    disparity = stereo.compute(img_1, img_2)
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min)) 
    return disparity


result = ShowDisparity(bSize=5) 
print(result) 
plt.imshow(result, 'gray') 
cv2.imwrite("output_p_6.png", result)
plt.axis('off')
plt.show()

