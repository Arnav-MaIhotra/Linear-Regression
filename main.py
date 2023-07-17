import numpy as np
import random
import regresspy
from sklearn import linear_model
    

X = np.array([[i] for i in range(-100, 100)])
y = np.array([2*i + 18 + random.uniform(-1, 1) for i in range(-100, 100)])

model = regresspy.LinearRegression(max_iters = 100000, tolerance = 1e-10)

sklearn_model = linear_model.LinearRegression()

sklearn_error = 0
error = 0

for i in range(100):

    sklearn_model.fit(X, y)

    model.fit(X, y)

    res = model.predict(4)

    sklearn_res = sklearn_model.predict(np.array([4]).reshape(1, -1))

    print("SciKit Learn: ", sklearn_res[0])
    print("Result: ", res[0])

    sklearn_error += abs(sklearn_res[0]-26)/26
    error += abs(res[0]-26)/26

print("SciKit Learn error: ", sklearn_error)
print("Error: ", error) #MINE BEAT IT!!
#SciKit Learn error:  0.24283563760386595
#Error:  0.24256012726330822
