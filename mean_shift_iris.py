####   Mean Shift  ####
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.cluster import MeanShift
ms = MeanShift()

#ms #check which parameters were used. It should be:
#MeanShift(bandwidth=None, bin_seeding=False, cluster_all=True, min_bin_freq=1, n_jobs=1, seeds=None)

ms.fit(iris.data)

#ms.labels_  #check the outcome

import pylab as pl
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
pl.figure('Mean shift finds 2 clusters')
for i in range(0, pca_2d.shape[0]):
    if ms.labels_[i] == 1:
        c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')
    elif ms.labels_[i] == 0:
        c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
pl.legend([c1,c2], ['Cluster 1', 'Cluster 2'])
pl.title('Mean shift finds 2 clusters')
pl.show()
