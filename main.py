import numpy as np
import random
from regresspy import LinearRegression
    

X = np.array([[i] for i in range(-50, 51)])
y = np.array([2*i + 1 + random.uniform(-1, 1) for i in range(-50, 51)])

model = LinearRegression()

model.fit(X, y, True)

print(model.predict(10))