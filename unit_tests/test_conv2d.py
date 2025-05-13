import numpy as np
from modules.conv2d import Conv2D


def test_conv2d():
    # Parameters
    img_width = 5
    img_height = 5
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2
    conv_algo = 0  # Assuming conv_algo is not used in this test

    # Initialize layer
    conv = Conv2D(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding,conv_algo=conv_algo)
    
    # Input: 1 image, 1 channel, 5x5 values from 0 to 24
    input_image = np.arange(img_height*img_height*in_channels*batch_size, dtype=np.float32).reshape(batch_size, in_channels, img_width, img_height)

    # Set all kernels to 1 and biases to 0
    conv.kernels = np.ones((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
    conv.biases = np.zeros(out_channels, dtype=np.float32)

    # Pad the input manually for expected output
    padded = np.pad(input_image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Compute expected output manually
    out_h = (padded.shape[2] - kernel_size) // stride + 1
    out_w = (padded.shape[3] - kernel_size) // stride + 1
    expected_output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    patch = padded[b, 0, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    expected_output[b, c, i, j] = np.sum(patch)  # kernel is all ones

    # Run the actual forward pass
    output = conv.forward(input_image)

    # Validate
    assert np.allclose(output, expected_output), "Conv2D (padding+stride) forward mismatch!"
    print("✅ Conv2D forward with padding and stride passed!")

test_conv2d()


