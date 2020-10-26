import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Processing import KMeans as mLM

np.random.seed(42)



def segmentImg():
    img = cv2.imread('images\dog.jpg',0)
    i2 = cv2.imread('images\dog.jpg')
    i3 = i2.reshape((i2.shape[0]*i2.shape[1],3))
    FB = mLM.makeLMfilters()
    imgbank = []
    for kernal in FB:
        res = cv2.filter2D(img,-1,kernal)
        res1d = res.flatten()
        np.abs(res1d)
        imgbank.append(res1d)
    imgbank = np.stack(imgbank,-1)
    kmeansres = KMeans(n_clusters=7)
    kmeansres.fit(imgbank)
    centers = kmeansres.cluster_centers_
    clus = np.reshape(kmeansres.labels_, img.shape)
    #norm_image = cv2.normalize(clus, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(clus)
    plt.show()
    np.savetxt("dang.txt",clus)

    trans = transferImg([0,5,2], clus, 'images/dog.jpg', 'images/bg2.jpg')

    plt.figure(1)
    plt.clf()
    plt.imshow(trans)
    plt.show()




def transferImg(fgs, idxImg, sImgFilename, tImgFilename):
    # Read the images, estimate their dimensions
    sImg = cv2.imread(sImgFilename)
    tImg = cv2.imread(tImgFilename)
    rows, cols, clrs = sImg.shape

    # Transfer idx onto tImg
    for r in range(rows):
        for c in range(cols):
            if idxImg[r, c] in fgs:
                tImg[r, c, 0] = sImg[r, c, 0]
                tImg[r, c, 1] = sImg[r, c, 1]
                tImg[r, c, 2] = sImg[r, c, 2]

    print('***transfer done')
    return tImg

segmentImg()