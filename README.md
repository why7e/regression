# regression
Applying simple regression methods in numPy. Each implementation is to have their own respective MODEL-example.py.

# ols.py
Multple Linear Regression implementation

Usage:
```
from ols import OLS

m = OLS()
m.fit(X, y)

pred_y = m.predict(new_X)
weights = m.get_weights()
```

# knn.py
K Nearest Neighbours Classification implementation

Usage:
```
from knn import KNN

m = KNN(k)
m.fit(X, y)

pred_y = KNN.predict(new_X)
```
