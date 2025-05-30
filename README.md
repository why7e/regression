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
intercept = m.get_intercept()
```

# ridge.py
Ridge (L2) normalised regression

Usage
```
from ridge import Ridge

m = Ridge()
m.fit(X, y)

pred_y = m.predict(new_X)
weights = m.get_weights()
intercept = m.get_intercept()
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

# In Progress:
- LASSO Regression
  A bit tougher as there's no closed solution - need to implement an optimisation algorithm (like coordinate descent)
