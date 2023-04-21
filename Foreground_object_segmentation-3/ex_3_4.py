"""Using the OpenCV BackgroundSubtractorMOG2 class,
implement a background subtraction algorithm for the pedestrian dataset."""
import cv2
import numpy as np


def binarize(I):
    """Binarize image"""
    I = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return I


def grayscale(I):
    """Convert the image to grayscale"""
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


bgsub = cv2.createBackgroundSubtractorMOG2() # Create a background subtractor object
start = 300
IP = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % start))

for i in range(start + 1, 1100):
    I = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i))
    fgmask = bgsub.apply(I) # Apply the background subtraction algorithm
    bgsub.setVarThreshold(16)
    bgsub.setDetectShadows(False)
    bgsub.setHistory(100) # Number of frames to consider for background modeling
    bgsub.setComplexityReductionThreshold(0.001)

    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE)
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    TP_median = np.sum(np.logical_and(fgmask, gt)) # True positive
    TN_median = np.sum(np.logical_and(np.logical_not(fgmask), np.logical_not(gt))) # True negative
    FP_median = np.sum(np.logical_and(fgmask, np.logical_not(gt))) # False positive
    FN_median = np.sum(np.logical_and(np.logical_not(fgmask), gt)) # False negative

    accuracy_median = (TP_median + TN_median) / (TP_median + TN_median + FP_median + FN_median) # Accuracy

    print('Frame: %d, Accuracy (Median): %.2f' % (i, accuracy_median))

    cv2.imshow('Frame', fgmask)
    cv2.waitKey(1)

cv2.destroyAllWindows()