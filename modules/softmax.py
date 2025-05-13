import math
from modules.layer import Layer

import numpy as np

class Softmax(Layer):
    def forward(self, input, training=True):  # input: [batch_size x num_classes]
        input = np.array(input).astype(dtype=np.float32)  # Ensure input is float for numerical stability
        self.output = np.zeros_like(input,np.float32)

        for i, row in enumerate(input):
            max_val = np.max(row)
            exps = np.exp(row - max_val)
            self.output[i] = exps / np.sum(exps)

        return self.output

    def backward(self, grad_output, learning_rate=None):
        # Assuming softmax used with cross-entropy loss, so gradient is simplified
        return grad_output