
import math
import time
import numpy as np
import pickle
import os
from modules.eval import evaluate
import sys

# NO TOCAR, NO HACE FALTA TOCAR NADA DE ESTE FICHERO
def save_model(model, filename='model_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

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


def train(model, train_images, train_labels, epochs=10, batch_size=64, learning_rate=0.01,
          save_path='saved_models', resume=False, test_images=None, test_labels=None,augmentor=None):
    old_acc = 0.0
    no_improv = 0
    num_samples = len(train_images)
    epoch = 0


    if resume and os.path.exists(save_path):
        print(f"Resuming training from {save_path} ...")
        model.load_weights(save_path)
        old_acc, _ = evaluate(model, test_images, test_labels, save_path=save_path, load_model=False)
    else:
        print("Training from scratch.")

    print("Starting with accuracy of ", old_acc)

    for epoch in range(epochs):
        
        # Shuffle the data
        indices = np.random.permutation(len(train_images))
        train_labels = np.array(train_labels)[indices]  # <- fix
        train_images = train_images[indices]
        #train_labels = train_labels[indices]
        
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for i in range(0, num_samples, batch_size):
            batch_images = train_images[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            if augmentor is not None:
                batch_images = augmentor.augment_batch(batch_images)

            # Forward
            output = model.forward(batch_images, curr_iter=99,training=True)

            # Loss + gradient
            loss, grad = compute_loss_and_gradient(output, batch_labels)
            total_loss += loss

            # Accuracy
            batch_correct = sum(
                np.argmax(out) == np.argmax(label)
                for out, label in zip(output, batch_labels)
            )
            correct += batch_correct

            # Backward
            grad = model.backward(grad, learning_rate, curr_iter=99) 

            # Logging
            batch_num = i // batch_size + 1
            total_batches = (num_samples + batch_size - 1) // batch_size
            sys.stdout.write(
                f"\r[{batch_num}/{total_batches}] "
                f"Loss: {loss:.4f} | "
                f"Batch Acc: {100 * batch_correct / len(batch_images):.2f}%"
            )
            sys.stdout.flush()

        # Epoch summary
        duration = time.time() - start_time
        accuracy = correct / num_samples
        ips = num_samples / duration

        

        print(f"\nEpoch Summary - Loss: {total_loss:.4f} | Acc: {accuracy * 100:.2f}% | IPS: {ips:.2f}")
        
        eacc, _ = evaluate(model, test_images, test_labels, save_path=save_path, load_model=False)
        if eacc > old_acc:
            old_acc = eacc
            # Save weights
            model.save_weights(save_path)
            print(f"Model saved to {save_path}")
            no_improv = 0
        else:
            no_improv += 1
            if no_improv >= 5:
                print("Early stopping due to no improvement.")
                break



