"""KNN Background Subtraction"""
import cv2
import numpy as np


def binarize(I):
    I = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return I


def grayscale(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


knn = cv2.createBackgroundSubtractorKNN() # Create KNN background subtractor
start = 300
IP = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % start))

for i in range(start + 1, 1100):
    I = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i))
    fgmask = knn.apply(I) # Apply KNN background subtractor

    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE)
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    # Apply morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Create a structuring element
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # Apply morphology to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    TP_median = np.sum(np.logical_and(fgmask, gt))
    TN_median = np.sum(np.logical_and(np.logical_not(fgmask), np.logical_not(gt)))
    FP_median = np.sum(np.logical_and(fgmask, np.logical_not(gt)))
    FN_median = np.sum(np.logical_and(np.logical_not(fgmask), gt))

    accuracy_median = (TP_median + TN_median) / (TP_median + TN_median + FP_median + FN_median)

    print('Frame: %d, Accuracy (Median): %.2f' % (i, accuracy_median))

    cv2.imshow('Frame', fgmask)
    cv2.waitKey(1)

cv2.destroyAllWindows()