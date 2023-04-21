"""Read an image, display it, save it, print its shape, size and type"""
import cv2

I = cv2.imread('../mandrill.jpg')
cv2.namedWindow('Mandrill')
cv2.imshow('Mandrill', I)
cv2.imwrite('mand.png',I)
print(I.shape)
print(I.size)
print(I.dtype)
cv2.waitKey(0)
cv2.destroyAllWindows()