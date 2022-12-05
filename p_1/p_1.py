import cv2
import matplotlib.pyplot as plt

interest_img = cv2.imread("img_8.png")
print(interest_img)

crop_img = interest_img[714:2228,341:1420]

test_images = [
    "img_1.png",
    "img_2.png",
    "img_3.png",
    "img_4.png",
    "img_5.png",
    "img_6.png",
    "img_7.png",
    "img_9.png",
    "img_10.png",
    "img_8.png"
]

correlate = []
for img in test_images:
    testImg = cv2.imread(img)
    cropTestImg = testImg[714:2228,341:1420]
    plt.imshow(cropTestImg)
    plt.show()
    X = cropTestImg - crop_img
    ssd = sum(X[:]**2)
    correlate.append(ssd)

print(correlate)
cv2.imshow("image of interest",interest_img)
plt.imshow(crop_img)
plt.show()