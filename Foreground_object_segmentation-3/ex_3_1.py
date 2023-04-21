"""This script performs mean and median background subtraction using a buffer of images."""
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
N = 60
start = 300
IP = binarize(grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % start)))
YY = IP.shape[0]
XX = IP.shape[1]
iN = 0

# Initialize buffer
BUF = np.zeros((YY, XX, N), np.uint8)

# Loop over frames
for i in range(start + 1, 1100):
    I = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i)) # Read image
    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE) # Read ground truth
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
    BUF[:, :, iN] = I

    # Increment buffer counter
    iN += 1
    # Check if buffer is full
    if iN == N:
        iN = 0

    # Calculate buffer mean and median
    buffer_mean = np.mean(BUF, axis=2).astype(np.uint8)
    buffer_median = np.median(BUF, axis=2).astype(np.uint8)

    # Perform background subtraction
    diff_mean = cv2.absdiff(buffer_mean, I)
    diff_median = cv2.absdiff(buffer_median, I)

    # Binarize the result
    thresh_mean = binarize(diff_mean)
    thresh_median = binarize(diff_median)

    # Apply median filter to object mask
    mask_mean = cv2.medianBlur(thresh_mean, 5)
    mask_median = cv2.medianBlur(thresh_median, 5)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_mean = cv2.morphologyEx(mask_mean, cv2.MORPH_OPEN, kernel)
    mask_median = cv2.morphologyEx(mask_median, cv2.MORPH_OPEN, kernel)

    # Calculate metrics
    TP_mean = np.sum(np.logical_and(mask_mean, gt))
    TN_mean = np.sum(np.logical_and(np.logical_not(mask_mean), np.logical_not(gt)))
    FP_mean = np.sum(np.logical_and(mask_mean, np.logical_not(gt)))
    FN_mean = np.sum(np.logical_and(np.logical_not(mask_mean), gt))

    TP_median = np.sum(np.logical_and(mask_median, gt))
    TN_median = np.sum(np.logical_and(np.logical_not(mask_median), np.logical_not(gt)))
    FP_median = np.sum(np.logical_and(mask_median, np.logical_not(gt)))
    FN_median = np.sum(np.logical_and(np.logical_not(mask_median), gt))

    # Calculate accuracy
    accuracy_mean = (TP_mean + TN_mean) / (TP_mean + TN_mean + FP_mean + FN_mean)
    accuracy_median = (TP_median + TN_median) / (TP_median + TN_median + FP_median + FN_median)

    # Display results
    cv2.imshow('Mean Method', mask_mean)
    cv2.imshow('Median Method', mask_median)
    cv2.waitKey(1)

    print('Frame: %d, Accuracy (Mean): %.2f, Accuracy (Median): %.2f' % (i, accuracy_mean, accuracy_median))

cv2.destroyAllWindows()