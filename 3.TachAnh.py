import matplotlib.image as mping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mping.imread('girl3.jpg')
plt.imshow(img)
#imgplot = plt.imshow(img) #? for what
plt.axis('off') #ko de kich thuoc
plt.show()

X = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

for K in [3,5]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img1 = np.zeros_like(X)
    for k in range(K):
        img1[label == k] = kmeans.cluster_centers_[k]
    img2 = img1.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img2, interpolation='nearest')
    plt.axis('off')
    plt.show()