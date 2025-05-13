from modules.layer import Layer



import numpy as np

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x, training=True):
        self.input = np.array(x, dtype=np.float32)  # ensure NumPy array
        return np.maximum(0, self.input)

    def backward(self, grad_output, learning_rate=None):
        return grad_output * (self.input > 0)