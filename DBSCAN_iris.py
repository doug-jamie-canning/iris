#### Unsupervised learning with DBSCAN (Density Based Spatial Clustering of Applications with Noise ####
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()

#dbscan  #check which parameters were used by typing dbscan
#DBSCAN (algorithm='auto', eps=0.5, leaf_size=30, metric='euclidean', min_sample=5, p=None, random_state=None)

dbscan.fit(iris.data)

#dbscan.labels_ # to check the outcome of the fit

#visualize the clusters
import pylab as pl
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
for i in range(0, pca_2d.shape[0]):
  if dbscan.labels_[i] == 0:
    c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
  elif dbscan.labels_[i] == 1:
    c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
  elif dbscan.labels_[i] == -1:
    c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
    pl.figure('DBSCAN finds 2 clusters and noise')
pl.legend([c1,c2,c3], ['Cluster 1', 'Cluster 2', 'Noise'])
pl.title('DBSCAN finds 2 clusters and noise')
pl.show()
