
import math
import time
import numpy as np

# NO TOCAR, NO HACE FALTA TOCAR NADA DE ESTE FICHERO
def compute_loss_and_gradient(predictions, labels):
    batch_size = len(predictions)
    loss = 0.0
    grad = []

    for pred, label in zip(predictions, labels):
        sample_loss = 0.0
        sample_grad = []
        for p, y in zip(pred, label):
            # Add small epsilon for numerical stability
            epsilon = 1e-9
            p = max(min(p, 1 - epsilon), epsilon)
            sample_loss += -y * math.log(p)
            sample_grad.append(p - y)
        loss += sample_loss
        grad.append(sample_grad)

    loss /= batch_size
    return loss, grad






def perf(model, train_images, train_labels, batch_size=64):
    num_samples = batch_size
    i=0
    batch_images = train_images[i:i+batch_size]

    start_time = time.time()
        
    output = batch_images
           
    output = model.forward(batch_images, curr_iter=i,training=False)
    
    duration = time.time() - start_time
    ips = num_samples / duration

    print(f"Total time: {duration:.2f}s IPS: {ips:.2f}images/sec")

    