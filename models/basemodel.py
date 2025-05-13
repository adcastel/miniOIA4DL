import time
import os
import numpy as np
class BaseModel:
    def __init__(self, layers):
        self.layers = layers
    
    def get_model(self):
        return self.layers

    def forward(self, x, curr_iter=1,training=False):
        imgs=x.shape[0]
        if curr_iter == 0:
            print("FW Layer;Batch;Time(s);Performance(imgs/s)")
        for layer in self.layers:
            layer_start_time = time.time()
            x = layer.forward(x)
            layer_time = time.time() - layer_start_time
            if curr_iter == 0:
                # Calculate performance metrics
                images_per_second = imgs / layer_time
                print(f"{layer.__class__.__name__};{imgs};{layer_time:.4f};{images_per_second:.2f}")
        if curr_iter == 0:
            print("==========================================")
        
        return x

    def backward(self, grad_output, learning_rate,curr_iter=1):
        imgs=len(grad_output)
        if curr_iter == 0:
            print("BW Layer;Batch;Time(s);Performance(imgs/s)")
        for layer in reversed(self.layers):
            layer_start_time = time.time()
            grad_output = layer.backward(grad_output, learning_rate)
            layer_time = time.time() - layer_start_time
            if curr_iter == 0:
                if layer_time == 0.0:
                    layer_time = 1e-10
                images_per_second =  imgs/ layer_time
                print(f"{layer.__class__.__name__};{imgs};{layer_time:.4f};{images_per_second:.2f}")
        if curr_iter == 0:
            print("==========================================")
        return grad_output

    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_weights'):
                np.savez(os.path.join(path, f'layer_{i}.npz'), **layer.get_weights())

    def load_weights(self, path):
        for i, layer in enumerate(self.layers):
            weight_path = os.path.join(path, f'layer_{i}.npz')
            if hasattr(layer, 'set_weights') and os.path.exists(weight_path):
                data = np.load(weight_path)
                layer.set_weights({k: data[k] for k in data.files})