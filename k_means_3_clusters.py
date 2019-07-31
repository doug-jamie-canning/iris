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
pl.figure('Iris dataset without labels as seen by K-means')
pl.scatter(pca_2d[:,0], pca_2d[:,1], c='black')
pl.title('Iris dataset without labels as seen by K-means')
pl.show()

pl.figure('K-means clusters the Iris dataset into 3 clusters')
for i in range(0, pca_2d.shape[0]):
  if kmeans.labels_[i] == 1:
    c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
  elif kmeans.labels_[i] == 0:
    c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
  elif kmeans.labels_[i] == 2:
    c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
pl.legend([c1,c2,c3], ['Cluster 1', 'Cluster 0', 'Cluster 2'])
pl.title('K-means clusters the Iris dataset into 3 clusters')
pl.show()
