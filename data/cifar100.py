import os
import pickle
import urllib.request
import tarfile
import numpy as np

def download_and_extract_cifar100(data_dir='cifar-100-python'):
    if not os.path.exists(data_dir):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        filename = url.split('/')[-1]
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        os.remove(filename)

def load_cifar100_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = dict[b'data']
    labels = dict[b'fine_labels']
    images = data.reshape(-1, 3, 32, 32).astype(np.float32)
    return images, labels

def load_cifar100(data_dir='cifar-100-python'):
    download_and_extract_cifar100(data_dir)
    train_images, train_labels = load_cifar100_batch(os.path.join(data_dir, 'train'))
    test_images, test_labels = load_cifar100_batch(os.path.join(data_dir, 'test'))

    # Ensure float32 for training, and convert labels to NumPy arrays
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels)
    
    return (train_images, train_labels), (test_images, test_labels)

def normalize_images(train_images,test_images):
    #mean = np.array([0.5071, 0.4867, 0.4408]).reshape(1, 3, 1, 1).astype(np.float32)
    #std = np.array([0.2675, 0.2565, 0.2761]).reshape(1, 3, 1, 1).astype(np.float32)
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    mean = np.mean(train_images, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.std(train_images, axis=(0, 2, 3), keepdims=True).astype(np.float32) + 1e-7
    
    # Normalize both datasets
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    return train_images, test_images

def one_hot_encode(labels, num_classes=100):
    one_hot = [[0] * num_classes for _ in range(len(labels))]
    for idx, label in enumerate(labels):
        one_hot[idx][label] = 1
    return one_hot
