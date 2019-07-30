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

For the files svm_iris_July_29_2019.py and svm_iris_July_29_2019.png
Using the iris dataset, a supervised learning model is created using support vector machine (SVM). A supervised learning model 
learns from the data that is already labeled. The dataset is split into a training set (90%) and a test set (10%). To visualize 
the classifier, the module matplotlib creates the plot. 

For the file random_forest_iris_July_29_2019.py
Using the iris dataset, a supervised learning model is created using an ensemble of decision trees to formulate a model. A random subset of the training data votes to select that best and strongest model. 

For the files k_means_iris_July_29_2019.py and k_means_iris_July_29_2019.png
Using the iris dataset, an unsupervised learning model is created using K-Means. K-Means is a clustering algorithm. It does not use a training dataset and it doesn't normally know the outcomes, therefore, the dataset isn't labeled and there is no acceptance of a target value during the creation of the clustering algorithm. K-Means is highly scalable, but is best used for datasets that have a smaller number of clusters with linearly separable data that is proportional in size.


