from __future__ import print_function
import numpy as np
from mnist import MNIST
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import cdist

data_set = MNIST('./MNIST/')
data_set.load_training()
train_img = data_set.train_images
train_img = np.array(train_img)/255.0
train_labels = data_set.train_labels
train_labels = np.array(train_labels)

data_set.load_testing()
test_img = data_set.test_images
test_labels = data_set.test_labels

test_img = np.array(test_img)[:1000]/255.0
test_labels = np.array(test_labels)[:1000]

D = cdist(test_img, train_img, metric='cosine')
print(D.shape)


labels = []
for it in range(5):
    max = np.argmin(D, axis=1)
    labels.append(train_labels[max])
    for i in range(1000):
        D[i][max] = 1

labels = np.array(labels).T
res = []
for it in range(labels.shape[0]):
    label = labels[it]
    weigh = np.array([5, 4, 3, 2, 1])
    max = 0
    lb = 0
    for i in range(5):
        sum = weigh[label == label[i]].sum()
        if sum > max:
            max = sum
            lb = label[i]
    res.append(lb)

res = np.array(res)
print("Accuracy of KNN: ", (100*accuracy_score(test_labels, res)))
