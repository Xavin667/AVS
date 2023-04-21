"""Simple thermal image analysis,added labeling and bounding box"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('vid1_IR.avi')

while cap.isOpened():
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (_, G) = cv2.threshold(G, 55, 255, cv2.THRESH_BINARY)
    m = cv2.medianBlur(G, 5)
    o = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3)))
    oc = cv2.morphologyEx(o, cv2.MORPH_CLOSE, np.ones((9,9)))


    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(oc)

    if stats.shape[0] > 1:  # are there any objects
        tab = stats[1:, 4]  # 4 columns without first element
        pi = np.argmax(tab)  # finding the index of the largest item
        pi = pi + 1  # increment because we want the index in stats, not in tab
        # drawing a bbox
        cv2.rectangle(oc, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0]+stats[pi, 2], stats[pi, 1] +
                                                         stats[pi, 3]), (255, 0, 0), 2)
        # print information about the field and the number of the largest element
        cv2.putText(oc, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(oc, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    cv2.imshow('IR', o)
    cv2.imshow('Labels', oc)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()