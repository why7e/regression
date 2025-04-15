import numpy as np

class OLS:
  def __init__(self, intercept=True):
    self.intercept = intercept

  # Fit an OLS model of paramaters A onto vector b
  def fit(self, A, b):
    self.A = A
    self.b = b

    if self.intercept:
      A = np.column_stack((np.ones((len(A), 1)), A))

    # Find the least-squares solution to the linear system Ax = b
    self.x = np.linalg.lstsq(A, b, rcond=None)[0]

  # Predict the value of b given a set of parameters A
  def predict(self, A):
    return A @ self.x

  # Get the weights of the model x
  def get_weights(self):
    if self.intercept:
      return self.x[1:]
    else:
      return self.x
  
  # Get the constant term of the model
  def get_intercept(self):
    if self.intercept:
      return self.x[0]
    else:
      return 0
