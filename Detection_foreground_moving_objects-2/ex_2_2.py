"""Performing filtering using erosion and dilation, as well as labeling and calculating parameters of
the detected objects. Implementing the computation of the metrics Precision, Recall and F1-score."""
import cv2
import numpy as np

f = open('../pedestrian/temporalROI.txt', 'r')  # open file
line = f.readline()  # read line
roi_start, roi_end = line.split()  # split line
roi_start = int(roi_start)  # conversion to int
roi_end = int(roi_end)  # conversion to int


def binarize(I):
    I = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return I


def grayscale(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


TP = 0
TN = 0
FP = 0
FN = 0

start = 300
IP = cv2.imread('../pedestrian/input/in000300.jpg', cv2.IMREAD_GRAYSCALE)

for i in range(roi_start, roi_end + 1):
    I_ORG = cv2.imread('../pedestrian/input/in%06d.jpg' % i)  # read the image
    gt = cv2.imread('../pedestrian/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE)  # read the ground truth
    (_, gt) = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)  # binarization of the ground truth
    I = grayscale(I_ORG)

    r = cv2.absdiff(I, IP)  # absolute difference between the current image and the first image
    (t, rb) = cv2.threshold(r, 15, 255, cv2.THRESH_BINARY)
    m = cv2.medianBlur(rb, 7)  # median filter
    o = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3)))
    oc = cv2.morphologyEx(o, cv2.MORPH_CLOSE, np.ones((9,9)))
    w = oc  # the result of the processing is the final mask

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(w)  # labeling of connected components

    I_VIS = I_ORG
    if stats.shape[0] > 1:  # are there any objects
        tab = stats[1:, 4]  # 4 columns without first element
        pi = np.argmax(tab)  # finding the index of the largest item
        pi = pi + 1  # increment because we want the index in stats, not in tab
        # drawing a bbox
        cv2.rectangle(I_VIS, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0]+stats[pi, 2],
            stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
        # print information about the field and the number of the largest element
        cv2.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(I_VIS, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    TP_M = np.logical_and((w == 255), (gt == 255))  # true positive
    TP_S = np.sum(TP_M)  # sum of true positive
    TP = TP + TP_S  # sum of all true positive

    TN_M = np.logical_and((w == 0), (gt == 0))
    TN += np.sum(TN_M)

    FP_M = np.logical_and((w == 255), (gt == 0))
    FP += np.sum(FP_M) 

    FN_M = np.logical_and((w == 0), (gt == 255))
    FN += np.sum(FN_M) 

    cv2.imshow('I_ORG', I_VIS)
    cv2.imshow('I', w)
    cv2.imshow('GT', gt)

    cv2.waitKey(10)