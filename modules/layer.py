import numpy as np
import os
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError
    
