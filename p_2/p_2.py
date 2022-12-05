import cv2
import numpy as np

file ="ab.mov"
capture = cv2.VideoCapture(file)
def img_alignment(img, img1):
    img, img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
    size_img = img.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img, img1, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img1, warp_matrix, (size_img[1], size_img[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(img1, warp_matrix, (size_img[1], size_img[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

while True:
    _, img1 = capture.read()
    _, img2 = capture.read()

    diff = cv2.absdiff(img1, img2)
    
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

    _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, b, l = cv2. boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            cv2.rectangle(img1, (x, y), (x+b, y+l), (0,255,0), 2)
    
    cv2.imshow("Motion", img1)
    input = cv2.waitKey(1)
    if input%256 == 27:
        print("Closing program")
        exit()