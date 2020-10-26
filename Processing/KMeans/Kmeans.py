import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def kmeans(kval,data):
    """
    general kmeans function creating kval clusters based on data, a 3 dimensional pandas dataframe
    :param kval: k clusters to find
    :param data: the 3 column pandas dataframe
    :return: the sse value calculated from the data, can also display a 3d scatterplot of the clusters if plts
            uncommented
    """
    #sets up max and min values for each dimension for start points
    zerocolumnrange = (data[0].min(),data[0].max())
    onecolumnrange = (data[1].min(),data[1].max())
    twocolumnrange = (data[2].min(),data[2].max())
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #renames the kval attribute to k
    k = kval
    #creates an array to hold the k random startpoints
    starts = np.zeros((k,3))
    #create an array to hold prev centers to check difference
    prevstarts = np.zeros((k,3))
    #sets up lists to hold the points assign to each k clusters
    clustersets = []
    #sets up start points and the clustersets
    for i in range(0,k):
        clustersets.append([])
        starts[i][0] = random.uniform(zerocolumnrange[0],zerocolumnrange[1])
        starts[i][1] = random.uniform(onecolumnrange[0],onecolumnrange[1])
        starts[i][2] = random.uniform(twocolumnrange[0],twocolumnrange[1])
    stopping_cond = True
    #ax.scatter(data.values.T[0],data.values.T[1],data.values.T[2], marker='o')
    #ax.scatter(starts[:,0],starts[:,1],starts[:,2], marker="v")
    #ax.set_xlabel('x0')
    #ax.set_ylabel('y1')
    #ax.set_zlabel('z2')
    #plt.show()
    prevcenter = np.zeros(len(data.values))
    noisereduction = False

    """
    Main clustering loop, runs until clusters only change slightly
    """
    while stopping_cond:
        #resets cluster sets
        clustersets = []
        for i in range(0, k):
            clustersets.append([])

        """
        loop over all data points and assigns them to a cluster
        """
        for row in data.iterrows():

            #skips the row if it has been marked as noise
            if prevcenter[row[0]] != 0:
                continue
            #setup the current row as a 3d coord
            cur_vector = np.zeros(3)
            cur_vector[0] = row[1][0]
            cur_vector[1] = row[1][1]
            cur_vector[2] = row[1][2]

            """
            Calculates mahalanobis distance
            """
            cur_minus = cur_vector - np.mean(cur_vector)
            cov = np.cov(data.values.T)
            inv_cov = sp.linalg.inv(cov)
            distlist = np.zeros(k)

            #loops over all k centers to find which minimizes mahalanobis
            for centpoint in range(0,k):
                dmeasure = distance.mahalanobis(cur_vector,starts[centpoint],inv_cov)
                distlist[centpoint] = abs(dmeasure)
            center_k = np.argmin(distlist)

            """
            if the noise reduction condition is met, the distance is checked to see if less than 3 mahalanobis units
            """
            if noisereduction:
                if distlist[center_k] < 3:
                    clustersets[center_k].append(cur_vector)
                else:
                    prevcenter[row[0]] += 1
            else:
                #sets this row to skip in future rows
                clustersets[center_k].append(cur_vector)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        symset = ['v','*','o','+','h']

        """
        Calculates center of mass and set that as new cluster center
        """
        if noisereduction:
            for center_ind in range(0,k):
                centroid = np.zeros(3)
                curn = 0
                xcord = []
                ycord = []
                zcord = []
                for p in clustersets[center_ind]:

                    xcord.append(p[0])
                    ycord.append(p[1])
                    zcord.append(p[2])
                    curn += 1
                    centroid[0] += p[0]
                    centroid[1] += p[1]
                    centroid[2] += p[2]
                if curn == 0:
                    curn += 1
                centroid[0] = centroid[0]/curn
                centroid[1] = centroid[1]/curn
                centroid[2] = centroid[2]/curn
                prevstarts[center_ind] = starts[center_ind]
                starts[center_ind] = centroid
                ax.scatter(xcord,ycord,zcord, marker=symset[center_ind])
                ax.scatter(centroid[0],centroid[1],centroid[2], marker=symset[center_ind], s=20*4)
        """
        STOPPING CONDITION
        """
        noisereduction = True
        ax.set_xlabel('x0')
        ax.set_ylabel('y1')
        ax.set_zlabel('z2')
        plt.show()
        diffcount = k//2

        """
        if only a few centers have moved by less than .1 on average, stop kmeans
        """
        for i in range(0,k):
            centmovement = prevstarts[i] - starts[i]
            movement = abs(centmovement[0]) + abs(centmovement[1]) + abs(centmovement[2])
            if movement < 1:
                diffcount -= 1
        if diffcount <= 0:
            stopping_cond = False
