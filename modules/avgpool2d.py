import numpy as np
from modules.layer import Layer


class GlobalAvgPool2D(Layer):

    def __init__(self):
        self.input = None

    def forward(self, x, training=True):  # shape: [batch, channels, h, w]
        self.input = x
        return np.mean(x, axis=(2, 3), keepdims=False).astype(np.float32)  # shape: [batch, channels]

    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, h, w = self.input.shape
        grad = grad_output[:, :, None, None] / (h * w)  # shape: [batch, channels, 1, 1]
        return np.ones_like(self.input) * grad  # broadcast to [batch, channels, h, w]