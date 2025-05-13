from modules.layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            # Create dropout mask: keep rate is (1 - p)
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x * self.mask / (1.0 - self.p)  # Scale to keep expected value
        else:
            return x  # No dropout during inference

    def backward(self, grad_output, learning_rate=None):
        return grad_output * self.mask / (1.0 - self.p)
#        else:
#            return grad_output

    