"""Histogram equalization"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

I = plt.imread('../mandrill.jpg')
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
plt.figure(1)


def hist(img):
    h = np.zeros((256, 1), np.float32)  # creates and zeros single-column arrays
    height, width = img.shape[:2]  # shape - we take the first 2 values

    for x in range(height):
        for y in range(width):
            i = img[x, y]
            h[i] = h[i] + 1
    return h


plt.figure(1)
plt.title('Histogram from function hist')
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist(IG))

# hist2
hist2 = cv2.calcHist([IG], [0], None, [256], [0, 256])

plt.figure(2)
plt.title('Histogram from function cv2.calcHist')
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist(IG))
plt.plot(hist2)

# equalization
IGE = cv2.equalizeHist(IG)
plt.figure(3)
plt.plot(hist(IGE))
plt.title('Equalized histogram using cv2.equalizeHist')
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# clahe equalisation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
I_CLAHE = clahe.apply(IG)
plt.figure(4)
plt.plot(hist(I_CLAHE))
plt.title('Equalized histogram using cv2.createCLAHE')
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.show()
cv2.imshow('Normal', IG)
cv2.imshow('CLAHE', I_CLAHE)
cv2.imshow('Basic equalization', IGE)
cv2.waitKey(0)