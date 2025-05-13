from modules.layer import Layer
import numpy as np

class BatchNorm2D(Layer):
    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((1, num_channels, 1, 1), dtype=np.float32)  # scale
        self.beta = np.zeros((1, num_channels, 1, 1), dtype=np.float32)  # shift

        # Running stats for inference
        self.running_mean = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_channels, 1, 1), dtype=np.float32)

    
    
    def forward(self, x, training=True):
        self.input = x

        if training:
            self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
            self.var = x.var(axis=(0, 2, 3), keepdims=True)
            
            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            out = self.gamma * self.norm + self.beta

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * norm + self.beta

        return out

    def backward(self, grad_output, learning_rate):
        B, C, H, W = grad_output.shape
        N = B * H * W

        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        
        # Gradients of parameters
        grad_norm = grad_output * self.gamma
        grad_var = np.sum(grad_norm * (self.input - self.mean) * -0.5 * std_inv**3, axis=(0, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + \
                    grad_var * np.mean(-2.0 * (self.input - self.mean), axis=(0, 2, 3), keepdims=True)
        
        # Gradient of input
        grad_input = grad_norm * std_inv + grad_var * 2.0 * (self.input - self.mean) / N + grad_mean / N

        # Update gamma and beta (SGD)
        grad_gamma = np.sum(grad_output * self.norm, axis=(0, 2, 3), keepdims=True)
        grad_beta = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)

        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta

        return grad_input

    def get_weights(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def set_weights(self, weights):
        self.gamma = weights["gamma"]
        self.beta = weights["beta"]
        self.running_mean = weights["running_mean"]
        self.running_var = weights["running_var"]