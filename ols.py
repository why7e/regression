import numpy as np

class OLS:
  # Fit an OLS model of paramaters A onto vector b
  def fit(self, A, b):
    self.A = A
    self.b = b

    # Find the least-squares solution to the linear system Ax = b
    self.x = np.linalg.lstsq(A, b, rcond=None)[0]

  # Predict the value of b given a set of parameters A
  def predict(self, A):
    return A @ self.x

  # Get the weights of the model x
  def get_weights(self):
    return self.x