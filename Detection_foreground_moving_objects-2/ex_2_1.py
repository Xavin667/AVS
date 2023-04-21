"""Image sequence loading and display"""
import cv2


def binarize(I):
    I = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return I


def grayscale(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


start = 300
IP = binarize(grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % start)))

for i in range(start + 1, 1100):
    I = binarize(grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i)))
    IC = cv2.absdiff(I, IP)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    IC = cv2.medianBlur(IC, 5)
    IC = cv2.erode(IC, kernel)
    IC = cv2.dilate(IC, kernel)
    
    IP = I
    cv2.imshow('I', IC)
    cv2.waitKey(10)