"""This scripts performs background substraction using approximation of mean (sigma-delta)"""
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

BGP.astype(np.float64) # Convert to float64
# Loop over frames
for i in range(start + 1, 1100):
    I = grayscale(cv2.imread('../pedestrian/input/in%06d.jpg' % i)) # Read image
    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE) # Read ground truth
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    BG = alfa * I + (1 - alfa) * BGP

    # Perform background subtraction
    diff = cv2.absdiff(np.uint8(BG), I)

    # Binarize the result
    thresh = binarize(diff)

    # Apply median filter to object mask
    mask = cv2.medianBlur(thresh, 5)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    # Calculate metrics
    TP_mean = np.sum(np.logical_and(mask, gt))
    TN_mean = np.sum(np.logical_and(np.logical_not(mask), np.logical_not(gt)))
    FP_mean = np.sum(np.logical_and(mask, np.logical_not(gt)))
    FN_mean = np.sum(np.logical_and(np.logical_not(mask), gt))

    # Calculate accuracy
    accuracy_mean = (TP_mean + TN_mean) / (TP_mean + TN_mean + FP_mean + FN_mean)
    # Display results
    cv2.imshow('Mean Method', mask)
    cv2.waitKey(1)

    print('Frame: %d, Accuracy (Mean): %.2f' % (i, accuracy_mean))

cv2.destroyAllWindows()