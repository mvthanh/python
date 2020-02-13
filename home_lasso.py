import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('home_data.csv', header=0)
#print((df))
df = np.asarray(df)
print(df.shape)
N = 500

#print(one)
#convert

one = np.ones((N, 1), dtype=np.float32)
x1 = np.array([df.T[0, :N]], dtype=np.float32).T
x2 = np.array([df.T[1, :N]], dtype=np.float32).T
x3 = np.array([df.T[4, :N]], dtype=np.float32).T
x4 = np.array([df.T[5, :N]], dtype=np.float32).T
x5 = np.array([df.T[6, :N]], dtype=np.float32).T
output = np.array([df.T[21, :N]], dtype=np.float32).T
#print(output)
input = np.concatenate((one, x1, x2, x3, x4, x5), axis=1)
#print(input)

A = np.dot(input.T, input)
#print(A)
b = np.dot(input.T, output)
#print(b)
AA = np.linalg.pinv(A)
#print(AA)
w = np.dot(AA, b)
print(w)

def cost(x):
    return w[0][0] + w[1][0]*x[1] + w[2][0]*x[2] + w[3][0]*x[3] + w[4][0]*x[4] + w[5][0]*x[5]

from sklearn.linear_model import Lasso
lasso = Lasso().fit(input, output)
print("Training set score: {:.2f}".format(lasso.score(input, output)))
print("lasso.coef_: {}".format(lasso.coef_))

print("cost")
for k in range(20):
    print(cost(input[k]), " : ", output[k][0])
    print(lasso.predict(input)[k])

from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(input, output)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
print(np.linalg.norm(regr.coef_ - w.T))


from sklearn.linear_model import Lasso
lasso = Lasso().fit(input, output)
print("Training set score: {:.2f}".format(lasso.score(input, output)))
print("lasso.coef_: {}".format(lasso.coef_))
print(lasso.predict(input)[:20])