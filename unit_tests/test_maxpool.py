from modules.maxpool2d import MaxPool2D
import numpy as np

def test_maxpool2d_forward_numerical():
    # Input: batch=2, channels=2, height=4, width=4
    x = np.array([
        [  # Sample 1
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9,10,11,12],
             [13,14,15,16]],

            [[16,15,14,13],
             [12,11,10,9],
             [8,7,6,5],
             [4,3,2,1]]
        ],
        [  # Sample 2
            [[1, 1, 1, 1],
             [2, 2, 2, 2],
             [3, 3, 3, 3],
             [4, 4, 4, 4]],

            [[9,8,7,6],
             [5,4,3,2],
             [1,0,-1,-2],
             [-3,-4,-5,-6]]
        ]
    ], dtype=np.float32)

    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool.forward(x)

    expected_output = np.array([
        [
            [[6, 8],
             [14, 16]],

            [[16, 14],
             [8, 6]]
        ],
        [
            [[2, 2],
             [4, 4]],

            [[9, 7],
             [1, -1]]
        ]
    ], dtype=np.float32)

    assert output.shape == expected_output.shape, "MaxPool2D output shape mismatch"
    assert np.allclose(output, expected_output), "MaxPool2D forward output values mismatch"
    print("✅ MaxPool2D forward large batch/channel test passed.")


test_maxpool2d_forward_numerical()