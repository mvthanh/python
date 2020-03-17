from __future__ import print_function
import numpy as np
from mnist import MNIST
from reportlab.lib.units import cm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
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

test_img = np.array(test_img)[:1000]
test_labels = np.array(test_labels)[:1000]

img = test_img[1].reshape((28, 28))


fig = plt.figure()
ax = plt.axes(projection="3d")

x_points = []
y_points = []
z_points = []
list = test_img[test_labels == 0]
res = []
for it in list:
    img = it.reshape((28, 28))
    x = np.argwhere(img > 0)
    x_point = x[:, 0]
    y_point = x[:, 1]
    z_point = []
    id = 0
    for i in x:
        x_points.append(x_point[id])
        y_points.append(y_point[id])
        z_points.append(img[i[0]][i[1]])
        id += 1

x_points = np.array(x_points).reshape((-1))
y_points = np.array(y_points).reshape((-1))
z_points = np.array(z_points).reshape((-1))
ax.scatter3D(x_points, y_points, z_points, s=1, alpha=1, c=z_points)
plt.show()

'''
D = cdist(test_img, train_img, metric='cosine')
print(D.shape)

res = train_labels[np.argmin(D, axis=1)]


labels = []
for it in range(5):
    max = np.argmin(D, axis=1)
    labels.append(train_labels[max])
    for i in range(1000):
        D[i][max] = 100000

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

print(test_img[1].reshape((28, 28)))
'''