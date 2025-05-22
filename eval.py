import time
import sys 
import numpy as np
import os

# NO TOCAR, NO HACE FALTA
def evaluate(model, test_images, test_labels,save_path,load_model=True):
    if load_model:
        if os.path.exists(save_path):
            print(f"Loading model from {save_path} ...")
            model.load_weights(save_path)
        else:
            print("Model not found. Please train the model first.")
            return
    
    start_time = time.time()
    correct = 0
    total = len(test_images)

    for i in range(total):
        output = test_images[i:i+1]  # [1 x ...]
       
        output = model.forward(output, curr_iter=i,training=False)

        predicted = np.argmax(output[0])
        actual = np.argmax(test_labels[i])
        if predicted == actual:
            correct += 1

        # Mini progress bar
        if i % 100 == 0 or i == total - 1:
            sys.stdout.write(f"\rEvaluating: {i+1}/{total}")
            sys.stdout.flush()

    accuracy = correct / total
    duration = time.time() - start_time
    ips = total / duration

    print(f"\nEvaluation Results - Accuracy: {accuracy * 100:.2f}% | IPS: {ips:.2f}")
    return accuracy, ips