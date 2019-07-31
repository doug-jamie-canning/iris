# iris

As an exercise in learning python, machine learning, and predictive analytics, it is customary to write the equivalent of a 
"hello world" program using the Iris dataset. The Iris Flower dataset, introduced in 1936 by Ronald Fisher in his article, "The Use of
of Multiple Measurements in Taxonomic Problems" is a real multivariate dataset that consists of three classes of the Iris flower 
(Iris Versicolor, Iris Setosa, and Iris Virginica). 

The dataset has 150 instances in total, each of the classes (Iris Versicolor, Iris Setosa, and Iris Virginica) containing 50
instances. There are 4 features (attributes) per sample which are Sepal Length, Sepal Width, Petal Width, and Target Class/Label.
The data set is somewhat linearly separable among the three classes. The Versicolor and Virginica classes are not perfectly 
separated by a straight line, even though it is close. The Setosa class can be linearly separated from the Versicolor and Virginica
class. The dataset is an ideal candidate for classification analysis, but not as good for clustering analysis.

The examples used within this readme have been derived from the book, "Predictive Analytics for Dummies" by Anasse Bari, PhD, MOhamed Chaouchi and Tommy Chung.

For the files svm.py and svm.png
Using the iris dataset, a supervised learning model is created using support vector machine (SVM). A supervised learning model 
learns from the data that is already labeled. The dataset is split into a training set (90%) and a test set (10%). To visualize 
the classifier, the module matplotlib creates the plot. 

For the file random_forest.py
Using the iris dataset, a supervised learning model is created using an ensemble of decision trees to formulate a model. A random subset of the training data votes to select that best and strongest model. 

For the files k_means.py and k_means.png
Using the iris dataset, an unsupervised learning model is created using K-Means. K-Means is a clustering algorithm. It does not use a training dataset and it doesn't normally know the outcomes, therefore, the dataset isn't labeled and there is no acceptance of a target value during the creation of the clustering algorithm. K-Means is highly scalable, but is best used for datasets that have a smaller number of clusters with linearly separable data that is proportional in size.

For the files k_means_without_labels.py and k_means_without_labels.png
Using the iris dataset, an unsupervised learning model is created using K-Means. The scatter plot has a data reduction to two-dimensions using Principal Component Analysis (PCA). This exercise shows the student how data might look in different dimensions, so that one can gain experience in unsupervised learning with datasets. In this example, the Virginica and Versicolor classes are not easily identifiable clusters. 

For the files k_means_3_clusters.py and k_means_3_clusters.png
Using the iris dataset, an unsupervised learning model is created using K-Means. The scatter plot now has labels because the output is known.

For the files k_means_4_clusters.py and K_means_4_clusters.png
Using the iris dataset, an unsupervised learning model is created using K-Means. The scatter plot shows that the cluster on the right produces less meaningful results as the value of k goes up.

For the files k_means_2_clusters.py and k_means_2_clusters.png
Using the iris dataset, an unsupervised learning model is created using K-Means. The scatter plot has labels, but only 2 clusters. The success rate of the prediction is 67%. At first glance, the model seems reasonable, but the guess is wrong regarding the amount of clusters.

For the files DBSCAN_2_clusters_and_noise.py and DBSCAN.png
DBSCAN is a clustering algorithm used in unsupervised learning. There are two parameters that are tuned in the algorithm. The parameter eps is the maximum distance between two parameters for which the data points are considered to be in the same neighborhood. The min_samples parameter is the minimum amont of data points in a neighborhood to be considered a cluster.

For the files mean_shift.py and mean_shift.png
Mean shift is a clustering algorithm used in unsupervised learning. There is a tuning parameter called bandwidth. Bandwidth is similar to chosing a size of a round window that can include the data points in a cluster. It isn't required to specify a number of clusters.
