#######################   K Means Scatter Plot of data points   ############################
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(iris.data)

#kmeans.labels_  # if you want to see the clusters that the algorithm produces
#iris.data.shape  #if you want to see the shape of the data which should be (150,4)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)

#pca_2d.shape # if you want to see the shape of the data after dimension reduction which should be (150,2)

import pylab as pl

kmeans4 = KMeans(n_clusters=4, random_state=111)
kmeans4.fit(iris.data)
pl.figure('K-Means with 4 clusters')
for i in range(0, pca_2d.shape[0]):
    if kmeans4.labels_[i] == 1:
        c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif kmeans4.labels_[i] == 0:
        c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif kmeans4.labels_[i] == 2:
        c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
    elif kmeans4.labels_[i] == 3:
        c4 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='c', marker='^')
pl.legend([c1,c2,c3,c4], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
pl.title('K-Means clusters the Iris dataset into 4 clusters')
pl.show()
