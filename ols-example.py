import random
import numpy as np
from ols import OLS

N = 100000
B_1 = 1
B_2 = 2
B_3 = 3
B_4 = 4

X = np.random.randint(0, 10, (N, 4))
# Add a constant variance term with mean 5
y = np.random.randint(0, 10, size=N) + B_1 * X[:, 0] + B_2 * X[:, 1] + B_3 * X[:, 2] + B_4 * X[:, 3]

ols = OLS()
ols.fit(X, y)

# Should approximate [B_1, B_2, B_3, B_4]
print(ols.get_weights())

# Should approximate 5
print(ols.get_intercept())
