from modules.layer import Layer

import numpy as np

class Flatten(Layer):
    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input_shape = input.shape  # Save shape for backward
        return input.reshape(input.shape[0], -1)  # Flatten each sample in batch

    def backward(self, grad_output, learning_rate=None):
        return grad_output.reshape(self.input_shape)  # Reshape back to original 4D