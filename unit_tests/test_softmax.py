from modules.softmax import Softmax
import numpy as np

def test_softmax_forward():
    np.random.seed(0)
    softmax = Softmax()

    # Simulate logits: batch size = 3, num_classes = 5
    x = np.array([
        [2.0, 1.0, 0.1, -1.0, 3.0],
        [1.5, 0.5, 2.5, 0.0, -0.5],
        [-2.0, 3.0, 0.0, 1.0, -1.0]
    ], dtype=np.float32)

    output = softmax.forward(x)

    # Manually compute softmax for each row
    def manual_softmax(logits):
        logits = logits - np.max(logits)  # stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    expected = np.array([manual_softmax(row) for row in x], dtype=np.float32)

    assert output.shape == expected.shape, "Softmax output shape mismatch"
    assert np.allclose(output, expected, atol=1e-6), "Softmax numerical output mismatch"
    print("✅ Softmax forward numerical test passed.")

test_softmax_forward()