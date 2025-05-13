# miniOIA4DL: A Deep Learning Framework for OIA subject

miniOIA4DL is a deep learning framework designed for learning how to optimizize the processes of traininig and evaluating convolutional neural networks (CNNs) on the CIFAR-100 dataset. It provides implementations of popular CNN architectures, including AlexNet, TinyCNN, ResNet18, and a custom-designed OIANet model. The layers are initially implemented in a simple numpy-based design where only numpy arrays are employed.
The educative goal is to motivate the students to apply the theoretical optimization techniques to a DL framework. The framework is modular, allowing for easy customization and extension of layers and models.

## Features

- **Predefined Models**: Includes implementations of AlexNet, TinyCNN, ResNet18, and OIANet for CIFAR-100.
- **Custom Layers**: Modular implementation of layers such as `Conv2D`, `ReLU`, `BatchNorm`, `MaxPool2D`, `GlobalAvgPool2D`, `Dense`, `Flatten`, and `Softmax`.
- **Training and Evaluation**: Built-in support for training models and evaluating their performance.
- **Performance Analysis**: Tools to measure and analyze model performance.
- **Cython Optimization**: Support for cython modules is included but not implemented :).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adcastel/miniOIA4DL.git
   cd miniOIA4DL
   ```
2. Build the Cython modules (if any):
   ```bash
   cd cython_modules
   python setup.py build_ext --inplace
   cd ..
   ```
## Usage

The arguments for launching are:
- **--model**: Model to evaluate. Currently supported: `AlexNet`, `TinyCNN`, `OIANet`, `ResNet18` (default:`OIANet`)
- **--batch_size**: Batch size for training (default: 8)
- **--epochs**: Number of epochs for training (default: 10)
- **--learning_rate**: Learning rate for training (default: 0.01)
- **--performance**: Enable performance measurement
- **--eval_only**: Enable evaluation-only mode
- **--conv_algo**: Conv2d algorithm 0-direct, 1-im2col, 2-im2colfused (default: 0)


To evaluate a model performance:
```bash
python main.py --model <model_name> --batch_size <batch_size> 
```
## Current Models

The following models are currently supported:
- **AlexNet**: A classic CNN architecture for image classification.
- **TinyCNN**: A lightweight CNN designed for quick experimentation.
- **ResNet18**: A residual network with 18 layers.
- **OIANet**: A custom-designed model optimized for CIFAR-100.

### Directory Structure

The repository is organized as follows:
```
miniOIA4DL/
├── data/                # Scripts for downloading and preprocessing CIFAR-100
├── models/              # Model definitions (AlexNet, TinyCNN, ResNet18, OIANet)
├── layers/              # Custom layer implementations (Conv2D, ReLU, etc.)
├── cython_modules/      # Cython-optimized operations
├── train.py             # Script for training models
├── evaluate.py          # Script for evaluating models
├── utils/               # Utility functions (e.g., data loaders, metrics)
├── checkpoints/         # Directory to save and load model checkpoints
└── README.md            # Project documentation
```