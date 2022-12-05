import cv2
import matplotlib.pyplot as plt

def objectDetect(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(blur, 10, 100)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edge, kernel, iterations=1)

    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = img.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print("objects were found in this image.")
    plt.axis('off')
    plt.imshow(image_copy,cmap='gray', vmin=0, vmax=255)
    plt.show()
    
img1 = "1.png"
img2 = "2.png"
img3 = "3.png"
img4 = "4.png"
objectDetect(img1)
objectDetect(img2)
objectDetect(img3)
objectDetect(img4)
