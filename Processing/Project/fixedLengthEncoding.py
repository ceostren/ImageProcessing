import cv2
import numpy as np

def encoding(img,b=10):
    l = img.shape[0]
    w = img.shape[1]
    bchannelvals = np.zeros(26)
    gchannelvals = np.zeros(26)
    rchannelvals = np.zeros(26)
    weakencoding = np.zeros((l,w,3))
    for i in range(0,l):
        for j in range(0,w):
            bval = img[i][j][0]//b
            gval = img[i][j][1]//b
            rval = img[i][j][2]//b
            bchannelvals[bval]+=1
            gchannelvals[gval]+=1
            rchannelvals[rval]+=1
            weakencoding[i][j]=[bval*b,gval*b,rval*b]
    cv2.imwrite('encodedweaklenna.jpg',weakencoding)




if __name__ == '__main__':
    lenna = cv2.imread('lenna.jpg')
    l2 = cv2.imread('encodedweaklenna.jpg',0
    )
    cv2.imwrite("colorless2.jpg",l2)
    #lenna = np.uint8(lenna)
    #newim = np.zeros((100,100,3))
    #for i in range(0,100):
    #    for j in range(0,100):
    #        newim[i][j][0] = lenna[i+220][j+220][0]
    #        newim[i][j][1] = lenna[i + 220][j + 220][1]
    #        newim[i][j][2] = lenna[i + 220][j + 220][2]
    #cv2.imwrite('lennaslice.png',newim)
    encoding(lenna,50)

