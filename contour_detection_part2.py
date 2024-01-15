import cv2 as cv
import numpy as np

#img = cv.imread("cookie.jpeg")
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    bw = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #grad3 = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    grad3 = cv.Canny(bw, 60, 400)
    #cv.imshow("canny", grad3)
    c,h = cv.findContours(grad3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
    cv.drawContours(frame, c, -1, (255,255,255), 3)
    cv.imshow("final", frame)
    
    key = cv.waitKey(5)
    if key == 27:
        break



cv.waitKey(0)

