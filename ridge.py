import numpy as np

class Ridge:
  def __init__(self, fit_intercept=True):
    self.fit_intercept = fit_intercept
    self.intercept = 0

  def fit(self, A, b, alpha = 0.1, method = 1):
    if self.fit_intercept:
      A = np.column_stack((np.ones((len(A), 1)), A));

    if method == 1:
      x = self._method1(A, b, alpha)
    elif method == 2:
      x = self._method2(A, b, alpha)
    else:
      raise ValueError("Invalid method")

    # Store the weights and intercept
    if self.fit_intercept:
      self.x = x[1:]
      self.intercept = x[0]
    else:
      self.x = x

  def _method1(self, A, b, alpha):
    regMatrix = alpha * np.eye(A.shape[1])

    # We don't actually want to regularise the intercept term - Remove the
    #   regularisation from the first row
    if self.fit_intercept:
      regMatrix[0, 0] = 0

    LHS = A.T @ A + regMatrix
    RHS = A.T @ b
    
    return np.linalg.solve(LHS, RHS);

  def _method2(self, A, b, alpha):
    regMatrix = np.sqrt(alpha) * np.eye(A.shape[1])
    if self.fit_intercept:
      regMatrix[0, 0] = 0
    
    # It is possible to define a new augmented system that allows us to use .lstsq
    # This effectively adds dummy data to pull the resultant x towards the zero vector.
    AMod = np.vstack([A, regMatrix])
    bMod = np.hstack([b, np.zeros(A.shape[1])])
    
    return np.linalg.lstsq(AMod, bMod, rcond=None)[0]

  def predict(self, A):
    return A @ self.x + self.intercept

  def get_weights(self):
    return self.x
  
  def get_intercept(self):
    return self.intercept
  