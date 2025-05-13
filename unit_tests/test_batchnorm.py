from modules.batchnorm import BatchNorm2D
import numpy as np


def test_batchnorm2d_forward_large():
    x = np.random.randn(8, 16, 32, 32).astype(np.float32)  # [batch=8, channels=16, h=32, w=32]
    bn = BatchNorm2D(16)
    bn.training = True
    out = bn.forward(x)

    assert out.shape == x.shape, "BatchNorm2D output shape mismatch"
    # Test mean ~0 and variance ~1 over batch+spatial dims
    flat = out.transpose(1, 0, 2, 3).reshape(16, -1)  # flatten per channel
    means = np.mean(flat, axis=1)
    vars_ = np.var(flat, axis=1)
    assert np.allclose(means, 0, atol=1e-1), "BatchNorm2D output mean not ~0"
    assert np.allclose(vars_, 1, atol=1e-1), "BatchNorm2D output var not ~1"
    print("✅ BatchNorm2D forward (large input) test passed.")

test_batchnorm2d_forward_large()