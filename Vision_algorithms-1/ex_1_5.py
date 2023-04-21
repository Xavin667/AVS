"""Performing arithmetic operations on images"""
import cv2
import numpy as np

L = cv2.imread('../lena.png')
I = cv2.imread('../mandrill.jpg')
cv2.imshow('Mandrill', I)
cv2.imshow('Lena', L)

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) #converts colours to gray
LG = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY) #converts colours to gray
cv2.imshow('Mandrill gray', IG)
cv2.imshow('Lena gray', LG)
cv2.imshow('Lena unit8', np.uint8(LG))
cv2.imshow('Lena abs', np.abs(LG))
cv2.imshow('Lena substraction', np.subtract(LG, 10))
cv2.imshow('Lena add', np.add(LG, 100))
print(IG.shape)
print(LG.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()