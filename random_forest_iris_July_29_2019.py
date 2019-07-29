#########################################Supervised Learning Model with Random Forest##########################################
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=15, random_state=111)
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.10, random_state=111)
rf = rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
predicted

######################Evaluate the model########################################
from sklearn import metrics
predicted
y_test
metrics.accuracy_score(y_test, predicted)
predicted == y_test

# Change the n_estimators parameter to 150
rf = RandomForestClassifier(n_estimators=150, random_state=111)
rf = rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
predicted
