import random
import numpy as np
from ridge import Ridge

N = 1000
B_0 = 8
B_1 = 10
B_2 = -5
B_3 = 3
B_4 = 7

# 4 Randomly generated covariates, with some correlation
X = np.random.randn(N, 4)
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(N)

y = B_0 + B_1 * X[:, 0] + B_2 * X[:, 1] + B_3 * X[:, 2] + B_4 * X[:, 3] + np.random.randn(N) * 2

print("True coefficients: [10, -5, 3, 7]")
print("True intercept: 8")
print()

# Test different regularization strengths
alphas = [0.0, 0.1, 1.0, 10.0, 100.0]

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Results for different alpha values:")
print("Alpha\tMethod\tWeights\t\t\t\tIntercept\tMSE")

for alpha in alphas:
    for method in [1, 2]:
        # Train model
        ridge = Ridge(fit_intercept=True)
        ridge.fit(X_train, y_train, alpha=alpha, method=method)
        
        # Get weights and intercept
        weights = ridge.get_weights()
        intercept = ridge.get_intercept()
        
        # Calculate MSE
        predictions = ridge.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        
        print(f"{alpha}\t{method}\t[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}, {weights[3]:.2f}]\t{intercept:.2f}\t{mse:.4f}")
    
    # Compare methods
    ridge1 = Ridge(fit_intercept=True)
    ridge2 = Ridge(fit_intercept=True)
    ridge1.fit(X_train, y_train, alpha=alpha, method=1)
    ridge2.fit(X_train, y_train, alpha=alpha, method=2)
    
    if np.allclose(ridge1.get_weights(), ridge2.get_weights()) and np.allclose(ridge1.get_intercept(), ridge2.get_intercept()):
        print(f"{alpha}\tMethods give same results")
    else:
        print(f"{alpha}\tMethods give different results")
    print()