"""This scripts performs background substraction using approximation of median (sigma-delta),
added conservative update approach"""
import numpy as np
import cv2


def binarize(I):
    """Binarize image"""
    I = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return I

def grayscale(I):
    """Convert image to grayscale"""
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


# Initialize buffer size and counter
start = 300
IP = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % start))
alfa = 0.01
BGP = IP
BGP.astype(np.float64)
# Loop over frames
for i in range(start + 1, 1100):
    I = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i))
    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE)
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    if (BGP < I).all():
        BG = BGP + 1
    if (BGP > I).all():
        BG = BGP - 1
    else:
        BG = BGP

    # Perform background subtraction
    diff = cv2.absdiff(np.uint8(BG), I)

    # Binarize the result
    thresh = binarize(diff)

    # Apply median filter to object mask
    mask = cv2.medianBlur(thresh, 5)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    bgmask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    BGP = cv2.multiply(alfa, I) + cv2.multiply(1 - alfa, BGP, bgmask)

    TP_median = np.sum(np.logical_and(mask, gt))
    TN_median = np.sum(np.logical_and(np.logical_not(mask), np.logical_not(gt)))
    FP_median = np.sum(np.logical_and(mask, np.logical_not(gt)))
    FN_median = np.sum(np.logical_and(np.logical_not(mask), gt))

    # Calculate accuracy
    accuracy_median = (TP_median + TN_median) / (TP_median + TN_median + FP_median + FN_median)
    # Display results
    cv2.imshow('Median Method', mask)
    cv2.waitKey(1)

    print('Frame: %d, Accuracy (Median): %.2f' % (i, accuracy_median))

cv2.destroyAllWindows()