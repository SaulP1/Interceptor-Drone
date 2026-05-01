import cv2, numpy as np
bgr = cv2.imread("C:\\Users\\gabeg\\Downloads\\hires_front_small_color_1777163499.png")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# Click-sample the balloon region
print(hsv[300:500, 300:700].reshape(-1,3).mean(axis=0))  # adjust crop to balloon