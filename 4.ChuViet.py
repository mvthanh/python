import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from display_network import *

mndata = MNIST('./MNIST/')
mndata.load_testing()
X = mndata.test_images
X0 = np.asarray(X)[:1000, :]/256.0
X = X0

K = 10
kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)
print(X)
print(pred_label)
N0 = 20
X1 = np.zeros((N0*K, 784))
X2 = np.zeros((N0*K, 784))

for k in range(K):
    Xk = X0[pred_label == k, :]

    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(N0).fit(Xk)
    dist, nearest_id  = neigh.kneighbors(center_k, N0)
    
    X1[N0*k: N0*k + N0,:] = Xk[nearest_id, :] #gan nhat
    X2[N0*k: N0*k + N0,:] = Xk[:N0, :] #ngau nhien

def display(X, K, N):
    plt.axis('off')
    A = display_network(X.T, K, N)
    f2 = plt.imshow(A, interpolation='nearest' )
    plt.gray()
    plt.show()

display(X1, K, N0)
display(X2, K, N0)