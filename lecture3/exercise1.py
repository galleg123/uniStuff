import copy
import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt





thresh = 0.01

img1 = cv2.imread('exercisefiles/aau-city-1.jpg')
img2 = cv2.imread('exercisefiles/aau-city-2.jpg')


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

dst1 = cv2.cornerHarris(gray1, 2, 3, thresh)
dst2 = cv2.cornerHarris(gray2, 2, 3, thresh)

dst1_pts = np.argwhere(dst1 > thresh)
dst2_pts = np.argwhere(dst2 > thresh)


# Draw the tested points on the original images
# img1[dst1>thresh] = [0,0,255]
# img2[dst2>thresh] = [0,0,255]


cv2.imshow("Img1", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = cv2.hconcat([img1, img2])




def sammenlign(img1, tuple1, img2, tuple2, k):
    y1 = tuple1[0]
    x1 = tuple1[1]
    y2 = tuple2[0]
    x2 = tuple2[1]

    batch1 = np.zeros((k*2+1, k*2+1))
    batch2 = np.zeros((k*2+1, k*2+1))
    
    if x1 - k < 0 or x1 + k >= img1.shape[1]: return None
    if y1 - k < 0 or y1 + k >= img1.shape[0]: return None
    if x2 - k < 0 or x2 + k >= img2.shape[1]: return None
    if y2 - k < 0 or y2 + k >= img2.shape[0]: return None


    batch1 = img1[y1-k:y1+k,x1-k:x1+k]
    batch2 = img2[y2-k:y2+k, x2-k:x2+k]

    diff = np.sum(cv2.absdiff(batch1, batch2))
    return diff


first = 1
for pt1 in dst1_pts:
    for pt2 in dst2_pts:
        diff = sammenlign(dst1, pt1, dst2, pt2,4)
        if diff == None: continue



        if diff <= thresh*3:
            # Save coordinates of the first one
            if first:
                first_pt1 = pt1
                first_pt2 = pt2
                first = 0
            # print(diff)
            # print(pt1, pt2)
            cv2.line(res, (pt1[1], pt1[0]), (pt2[1]+img1.shape[1], pt2[0]), (255, 0, 0), 1)


print("Points 1: ",first_pt1, " Points 2: ", first_pt2)


# Combining the images
new_image = np.zeros((img1.shape[0],img1.shape[1]*2,3), dtype=np.uint8)
new_image[0:new_image.shape[0],0:first_pt1[1]] = img1[0:img1.shape[0],0:first_pt1[1]]
new_image[0:new_image.shape[0],first_pt1[1]:first_pt1[1]+img2.shape[1]-first_pt2[1]] = img2[0:new_image.shape[0],first_pt2[1]:img2.shape[1]]



cv2.imshow("Compare", res)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Combined", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()