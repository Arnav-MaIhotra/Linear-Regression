import numpy as np
import random
import regresspy
from sklearn import linear_model
import time

X = np.array([[i] for i in range(-100, 100)])
y = np.array([2*i + 18 + random.uniform(-1, 1) for i in range(-100, 100)])

model = regresspy.LinearRegression(max_iters = 100000, tolerance = 1e-10)

sklearn_model = linear_model.LinearRegression()

sklearn_error = 0
error = 0

sktime = 0
mtime = 0

for i in range(10):
    st = time.time()
    sklearn_model.fit(X, y)
    et = time.time()

    sktime += et-st

    st = time.time()
    model.fit(X, y)
    et = time.time()

    mtime += et-st

    res = model.predict(4)

    sklearn_res = sklearn_model.predict(np.array([4]).reshape(1, -1))

    print("SciKit Learn: ", sklearn_res[0])
    print("Model: ", res[0])

    sklearn_error += abs(sklearn_res[0]-26)/26
    error += abs(res[0]-26)/26


print("SciKit Learn mean time: ", sktime/10)
print("Model mean time: ", mtime/10)
print("SciKit Learn error: ", sklearn_error*10)
print("Model error: ", error*10)

while True:
    inp = input(">>> ")
    if inp == "exit":
        break
    try:
        exec(inp)
    except Exception as e:
        print(e)
