import numpy as np

class OLS:
  def __init__(self, fit_intercept=True):
    self.fit_intercept = fit_intercept
    self.intercept = 0

  # Fit an OLS model of paramaters A onto vector b
  def fit(self, A, b):
    if self.fit_intercept:
      A = np.column_stack((np.ones((len(A), 1)), A))

    # Find the least-squares solution to the linear system Ax = b
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Store the weights and intercept
    if self.fit_intercept:
      self.x = x[1:]
      self.intercept = x[0]
    else:
      self.x = x
  
  # Predict the value of b given a set of parameters A
  def predict(self, A):
    return A @ self.x + self.intercept

  # Get the weights of the model x
  def get_weights(self):
    return self.x
  
  # Get the constant term of the model
  def get_intercept(self):
   return self.intercept
