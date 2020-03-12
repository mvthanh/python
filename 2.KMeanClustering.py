from __future__ import print_function
import numpy as np
import cv2
from mnist import MNIST
from scipy.spatial.distance import cdist


def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(new_centers, centers[-1]):
            break
        centers.append(new_centers)
        it += 1
    return centers, labels, it


data_set = MNIST('./MNIST/')
data_set.load_testing()
X = data_set.test_images
X = np.array(X)/255.0
(centers, labels, it) = kmeans(X, 10)
for it in range(10):
    image = np.array(centers[-1][it])
    image = image.reshape((28, 28, 1))
    cv2.imshow('image' + str(it), image)


def check_type(data):
    data = np.array(data).reshape(-1, 784)
    D = cdist(data, centers[-1])

    return np.argmin(D, axis=1)


data = X[2]
print(check_type(data))
img = np.array(data).reshape((28, 28, 1))
cv2.imshow("img", img)
cv2.waitKey(0)