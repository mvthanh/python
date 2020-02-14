import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('home_data.csv', header=0)
#print((df))
df = np.asarray(df)
print(df.shape)
N = 500

#print(one)
#convert
def pow2(x):
    xx = np.zeros(x.shape)
    for k in range(x.shape[0]):
        xx[k, 0] = x[k, 0]**2
    return xx
def dotxixj(x1, x2):
    xx = np.zeros(x1.shape)
    for k in range(x1.shape[0]):
        xx[k, 0] = x1[k, 0]*x2[k, 0]
    return xx

one = np.ones((N, 1), dtype=np.float32)
x1 = np.array([df.T[0, :N]], dtype=np.float32).T
x2 = np.array([df.T[1, :N]], dtype=np.float32).T
x3 = np.array([df.T[4, :N]], dtype=np.float32).T
x4 = np.array([df.T[5, :N]], dtype=np.float32).T
x5 = np.array([df.T[6, :N]], dtype=np.float32).T
x6 = pow2(x1)
x7 = pow2(x2)
x8 = pow2(x3)
x9 = pow2(x4)
x10 = pow2(x5)
x11 = dotxixj(x1, x2)
x12 = dotxixj(x1, x3)
x13 = dotxixj(x1, x4)
x14 = dotxixj(x1, x5)
x15 = dotxixj(x2, x3)
x16 = dotxixj(x2, x4)
x17 = dotxixj(x2, x5)
x18 = dotxixj(x3, x4)
x19 = dotxixj(x3, x5)
x20 = dotxixj(x4, x5)


output = np.array([df.T[21, :N]], dtype=np.float32).T
#print(output)
input = np.concatenate((one, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20), axis=1)
input1 = np.concatenate((one, x1, x2, x3, x4, x5), axis=1)
#print(input)

A = np.dot(input.T, input)
#print(A)
b = np.dot(input.T, output)
#print(b)
AA = np.linalg.pinv(A)
#print(AA)
w = np.dot(AA, b)
print(w)

def cost(w, x):
    r1 = w[0][0] + w[1][0]*x[1] + w[2][0]*x[2] + w[3][0]*x[3] + w[4][0]*x[4] + w[5][0]*x[5] + w[6][0]*(x[1]**2) + w[7][0]*(x[2]**2) + w[8][0]*(x[3]**2) + w[9][0]*(x[4]**2) + w[10][0]*(x[5]**2)
    r2 = w[11][0]*x[1]*x[2] + w[12][0]*x[1]*x[3] + w[13][0]*x[1]*x[4] + w[14][0]*x[1]*x[5] + w[15][0]*x[2]*x[3] + w[16][0]*x[2]*x[4] + w[17][0]*x[2]*x[5] + w[18][0]*x[3]*x[4] + w[19][0]*x[3]*x[5] + w[20][0]*x[4]*x[5] 
    return r1 + r2


from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(input, output)
  
print("cost")
for k in range(20):
#    print(cost(regr.coef_.T, input[k]), " : ", output[k][0])
    print(cost(w, input[k]), " : ", output[k][0])

def RegressionModel(input, output):
    lasso_linear = linear_model.LinearRegression(fit_intercept=False)
    # Training process
    lasso_linear.fit(input, output)
    # Evaluating the model
    return lasso_linear.score(input, output)

def polynomialRegression(X, Y):
    poly_model = Pipeline([('poly', PolynomialFeatures(2)),# tao lai bo du lieu moi theo bac 
                           ('linear', linear_model.LinearRegression(fit_intercept=False))]) #dung linear lai
    poly_model = poly_model.fit(X, Y)
    print(poly_model.predict(X)[:20].T)
    score_poly_trained = poly_model.score(X, Y)

    return score_poly_trained

print(RegressionModel(input, output))
print(polynomialRegression(input1, output))
