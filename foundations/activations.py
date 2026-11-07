import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # function = lambda x: np.round(1 / (1 + np.exp(-x)), 5)
        # new_z = function(z)
        new_z = np.round(1 / (1 + np.exp(-z)), 5)
        return new_z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # function = lambda x : np.maximum(0, x)
        # new_z = function(z)
        new_z = np.maximum(0, z)
        return new_z
