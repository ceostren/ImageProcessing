import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def imageProcessing():
    img = cv2.imread("lenna.jpg",0)
    r = img.shape[0]
    c = img.shape[1]
    patch = img[220:236,220:236]
    cv2.imwrite("test.jpg",patch)
    normpatch = normalize(patch)
    normimg = normalize(img)
    filterHelper(normimg,normpatch)

def gaussian1d():
    sigma = 1
    size = 3
    gf = np.zeros((1,3))
    for i in range(1,size+1):
        nfactor = 1/(math.sqrt(2*math.pi*(sigma**2)))
        efactor = math.exp((-i**2)/(2*sigma**2))
        gf[0][i-1] = nfactor*efactor
    return gf

def gaussian2d():
    sigma = 5
    size = 3
    gf = np.zeros((size,size))
    for i in range(1,size+1):
        for j in range(1, size+1):
            nfactor = 1/(math.sqrt(2*math.pi*(sigma**2)))
            efactor = math.exp((-(i**2 + j**2))/(2*sigma**2))
            gf[i-1][j-1] = nfactor*efactor
    return gf

def generateGuass():
    G1D = gaussian1d()
    print(G1D)
    G1Dt = G1D.reshape(G1D.shape+(1,))
    G2D1 = np.dot(G1D.T,G1D)
    print(G2D1)
    G2D2 = gaussian2d()
    print(G2D2)
    x = G2D2.max() - G2D2.min()
    G2D2 /= G2D2.sum()
    print(G2D2)
    img = cv2.imread("lenna.jpg",0)
    filterHelper(img,G2D2,"gausres.jpg")

def edgeDetection():
    img = cv2.imread("lenna.jpg",0)
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelx_lenna = filterHelper(img,sobelx,"sobelx_lenna.jpg")
    sobely_lenna = filterHelper(img,sobely,"sobely_lenna.jpg")

    prewittx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitty = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    prewittx_lenna = filterHelper(img,prewittx,"prewittx_lenna.jpg")
    prewitty_lenna = filterHelper(img, prewitty, "prewitty_lenna.jpg")
    prewittxy1_Lenna = filterHelper(prewittx_lenna,prewitty,"prewittxy1_lenna.jpg")
    prewittxy2_Lenna = filterHelper(prewitty_lenna,prewittx,"prewittxy2_lenna.jpg")

def cornerDetection(size,name):
    img = cv2.imread("Checkerboard.png",0)
    r = img.shape[0]
    c = img.shape[1]
    scoremap = np.zeros((r,c,3))
    f = size//2
    for i in range(f+1,r-f-1):
        for j in range(f+1,c-f-1):
            score = 0
            subarr = img[i-f:i+f+1,j-f:j+f+1].copy()
            sxn = (subarr - img[i-f:i+f+1,j-f-1:j+f].copy()).sum()
            sxp = (subarr - img[i - f:i + f + 1, j - f + 1:j + f + 2].copy()).sum()
            syn = (subarr - img[i-f-1:i+f,j-f:j+f+1].copy()).sum()
            syp = (subarr - img[i-f+1:i+f+2,j-f:j+f+1].copy()).sum()
            pdn = (subarr - img[i-f-1:i+f,j-f-1:j+f].copy()).sum()
            pdp = (subarr - img[i - f+1:i + f + 2, j - f+1:j + f + 2].copy()).sum()
            ndn = (subarr - img[i-f-1:i+f,j-f+1:j+f+2].copy()).sum()
            ndp = (subarr - img[i - f + 1:i + f + 2, j - f-1:j + f].copy()).sum()
            if sxn != sxp:
                score += 1
            if pdn != pdp:
                score += 1
            if syn != syp:
                score += 1
            if ndn != ndp:
                score += 1
            if score >= 4:
                scoremap[i][j][1] = 255
                #corner
            elif score >= 1:
                scoremap[i][j][2] = 255
            else:
                scoremap[i][j][0] = 255
    cv2.imwrite(name,scoremap)





#Function for cross correlation implementation
def filterHelper(img,temp,name):
    ir = img.shape[0]
    ic = img.shape[1]
    tr = temp.shape[0]
    tc = temp.shape[1]
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
    cv2.imwrite(name, final)
    return final

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

def harriscorner(size,name):
    img = cv2.imread("Checkerboard.png",0)
    r = img.shape[0]
    c = img.shape[1]
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    scoremap = np.zeros((r,c))
    f = size//2
    for i in range(1,r-1):
        for j in range(1,c-1):
            score = 0
            subarr = img[i-f:i+f+1,j-f:j+f+1].copy()
            resx = sobelx*subarr
            resy = sobely*subarr
            ix = resx.sum()
            iy = resy.sum()
            M = np.array([[ix*ix,ix*iy],
                          [iy*ix,iy*iy]])
            ev,rv = np.linalg.eig(M)
            R = ev[0]*ev[1] - .05*(ev[0] + ev[1])
            res = M - (1/R)*rv*(R)
            scoremap[i][j] = res.sum()


    cv2.imwrite(name,scoremap)

#imageProcessing()
generateGuass()
#edgeDetection()
#cornerDetection(5,"cornermap5x5.png")
harriscorner(3,"harriscorner.png")