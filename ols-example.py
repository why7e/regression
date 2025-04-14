import random
import numpy as np
from ols import OLS

B_1 = 1
B_2 = 2
B_3 = 3
B_4 = 4

X = np.random.randint(0, 10, (10, 4))
y = B_1 * X[:, 0] + B_2 * X[:, 1] + B_3 * X[:, 2] + B_4 * X[:, 3] + np.random.randint(0, 10)

print(X)
print(y)

ols = OLS()
ols.fit(X, y)
print(ols.get_weights())
