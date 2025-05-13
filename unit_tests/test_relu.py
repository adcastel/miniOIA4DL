from modules.relu import ReLU
import numpy as np

def test_relu_forward_numerical():
    np.random.seed(0)
    relu = ReLU()

    # Simulate input with batch size = 3, channels = 4, height = 2, width = 2
    x = np.array([
        [[[-1.0, 2.0], [3.0, -4.0]], [[1.0, -2.0], [-3.0, 4.0]], [[-1.5, 2.5], [3.5, -4.5]], [[0.5, -0.5], [-1.0, 1.0]]],
        [[[0.0, 1.0], [-2.0, 3.0]], [[-1.0, -2.0], [1.0, 4.0]], [[-2.5, 3.5], [-4.5, 5.5]], [[-0.5, 0.5], [1.0, -1.0]]],
        [[[3.0, -1.0], [0.5, -2.5]], [[-0.5, 2.5], [1.0, 2.0]], [[1.5, -2.0], [-3.0, 4.0]], [[2.0, 3.0], [-1.0, 0.0]]]
    ], dtype=np.float32)

    output = relu.forward(x)

    # Manually compute ReLU output
    expected = np.maximum(x, 0)

    assert output.shape == expected.shape, "ReLU output shape mismatch"
    assert np.allclose(output, expected, atol=1e-6), "ReLU numerical output mismatch"
    print("✅ ReLU forward numerical test passed.")

test_relu_forward_numerical()