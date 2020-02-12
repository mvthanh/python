import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print('Number of classes: ', len(np.unique(iris_y)))
print('Number of data points: ', len(iris_y)) 


"""X0 = iris_X[iris_y == 0,:]
print('Samples from class 0:')
print(X0[:5,:]) 

X1 = iris_X[iris_y == 1,:]
print('Samples from class 1:')
print(X1[:5,:])
X2 = iris_X[iris_y == 2,:]
print('Samples from class 2:')
print(X2[:5,:])"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print("Training size: ", len(y_train)) 
print("Test size    : ", len(y_test)) 

def myweight(distances):
    sigma = 0.5
    return np.exp(-distances**2/sigma)

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:") 
print("Predicted labels: ", y_pred[0:20]) 
print("Ground truth    : ", y_test[0:20]) 

from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: ", (100*accuracy_score(y_test, y_pred))) 