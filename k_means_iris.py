####################   K Means     ################################
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(iris.data)

#kmeans.labels_ #see the clusters that the algorithm produces
#iris.data.shape  #the current shape of the Iris datasets is (150,4)

from sklearn.decomposition import PCA  #dimensionality reduction
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)

#pca_2d.shape  #should be (150,2)

##################  plot the results ##################################
import matplotlib as mpl
mpl.use('WXAgg')
import matplotlib.pylab as pl
import numpy as np
pl.figure('K Means Model')
for i in range(0, pca_2d.shape[0]):
  if iris.target[i] == 0:
    c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
  elif iris.target[i] == 1:
    c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
  elif iris.target[i] == 2:
    c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')
pl.legend([c1,c2,c3], ['Setosa', 'Versicolor', 'Virginica'])
pl.title('Iris dataset with 3 clusters and known outcomes')
pl.show()
