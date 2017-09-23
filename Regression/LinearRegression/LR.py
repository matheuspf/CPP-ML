import numpy as np

from sklearn.linear_model import LinearRegression, BayesianRidge


X = np.array([[1, 2], [2, 4], [3, 7]])
y = np.array([1, 2, 10])

#lr = LinearRegression()
lr = BayesianRidge()

lr.fit(X, y)

#print(lr.coef_, lr.intercept_)

print(lr.predict(X))


"""
m = 100000
n = 100

X = np.random.rand(m, n)
y = np.random.rand(m)

lr = LinearRegression()

lr.fit(X, y)

z = lr.predict(X)
"""
