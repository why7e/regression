import numpy as np

# Where a, b are vectors, return the Euclidean distance between them
def dist(a, b):
  return np.sqrt(np.sum((a-b)**2))

class KNN:
  # Stubbing this because this seems hard
  def __init__(self, k):
    self.k = k
    return
  
  def fit(self, X, y):
    self.X = X
    self.y = y
    return
  
  def predict(self, z):
    X = self.x
    y = self.y
    k = self.k
    
    # Create an array to store distances and corresponding y values
    distances = np.zeros((len(X), 2))
    
    # Calculate distance from each training point to the query point z
    for i in range(len(X)):
      distances[i, 0] = dist(X[i], z)
      distances[i, 1] = y[i]
      
    # Sort X values by distance and get the k nearest y values
    distances = distances[distances[:, 0].argsort()]
    nearest_y = distances[:k, 1]
    
    # Return the most common class among the k nearest neighbors
    return np.bincount(nearest_y).argmax()