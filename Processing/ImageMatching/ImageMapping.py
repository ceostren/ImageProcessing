"""
Author Carl Ostrenga <ceo8099@rit.edu>
"""

import math

from Processing.ImageMatching import draw_patch
import numpy as np
import cv2


#Helper function to confirm q2 part 1
def q21():
    img = np.array([[2,1,6,-4],[0,-3,5,-2]])
    ker = np.array([[0,-1],[1,0]])
    print(np.matmul(ker,img))

#Helper function to confirm q2 part 2
def q22():
    img = np.array([[-2,3,-1],[3,-3,-2]])
    ker = np.array([[1,0],[0,-1]])
    print(np.matmul(ker,img))

#Main function which reads the images and calls the cross correlation and ssd functions
def patternRec():
    img = cv2.imread("waldo_onIce.png",0)
    template = cv2.imread("waldo_template.png",0)
    template = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    template = template * 1/2
    template = np.transpose(template)
    print(template.shape)
    print(img.shape)
    np.zeros((72+499-1,594+47-1))
    offr = 72
    offc = 47
    filterHelper(img,template)
    res = cv2.filter2D(img,-1,template)
    cv2.imshow('dingus',res)
    #ssdfilterHelper(img,template)


#Helper function which normalizes the image
def normalize(img):
    ir = img.shape[0]
    ic = img.shape[1]
    avg = img.mean()
    dst = np.zeros((ir, ic))
    for i in range(0, ir):
        for j in range(0, ic):
            dst[i][j] = (img[i][j] - avg) / (math.sqrt((img[i][j] - avg) ** 2))
    return dst


#Function for cross correlation implementation
def filterHelper(img,temp):
    ir = img.shape[0]
    ic = img.shape[1]
    tr = temp.shape[0]
    tc = temp.shape[1]
    #img = normalize(img)
    #temp = normalize(temp)
    #outi = np.zeros((ir+tr-1,ic+tc-1))
    final = img.copy()
    maxval = 0
    maxind = np.array([0,0])
    final = np.zeros((ir,ic))
    for i in range(0,ir-tr+1):
        for j in range(0,ic-tc+1):
            subarr = img[i:tr+i,j:j+tc]
            corrval = np.multiply(temp,subarr)
            corrval = corrval.sum()
            final[i+tr//2 + 1][j+tc//2 + 1] = corrval
            if corrval > maxval:
                maxval = corrval
                maxind = np.array([i+tr//2 +1,j+tc//2 + 1])
    #cv2.imwrite("corrMap.png" ,final)
    cv2.imwrite("lennamatch.png", final)
    #draw_patch.draw_patch(final2,maxind[0],maxind[1],7,3)
    #cv2.imwrite("normWaldo.png",final2)

#Function which implements a sum of square differences pattern matching
def ssdfilterHelper(img,temp):
    ir = img.shape[0]
    ic = img.shape[1]
    tr = temp.shape[0]
    tc = temp.shape[1]
    img = normalize(img)
    temp = normalize(temp)
    final = img.copy()
    final2 = cv2.imread("waldo_onIce.png")
    minval = 9999999999999999
    minind = np.array([0,0])
    final = np.zeros((ir,ic))
    for i in range(0,ir-71):
        for j in range(0,ic-46):
            subarr = np.zeros((tr,tc))
            subarr = img[i:tr+i,j:j+tc]
            corrval = 0
            corrval = ((temp-subarr)**2).sum()
            final[i+tr//2 + 1][j+tc//2 + 1] = corrval
            if corrval < minval:
                minval = corrval
                minind = np.array([i+tr//2 +1,j+tc//2 + 1])
    cv2.imwrite("ssdmap.png" ,final)
    draw_patch.draw_patch(final2, minind[0], minind[1], 7, 2)
    cv2.imwrite("ssdwaldo.png",final2)


patternRec()


