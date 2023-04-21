"""Colour space conversion using OpenCV"""
import cv2

I = cv2.imread('../mandrill.jpg')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # converts colours to gray
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)  # converts colours to HSV space
# cv2.imshow('Mandrill', I)  #display normal image
cv2.imshow('Mandrill IG', IG)  #display grayscale image
cv2.imshow('Mandrill IHSV', IHSV)  #display hsv image
IH = IHSV[:,:,0]
IS = IHSV[:,:,1]
IV = IHSV[:,:,2]
cv2.imshow('Mandrill IH', IH)  #display h image
cv2.imshow('Mandrill IS', IS)  #display s image
cv2.imshow('Mandrill IV', IV)  #display v image
print(I.size)
print(I.dtype)
cv2.waitKey(0)
cv2.destroyAllWindows()