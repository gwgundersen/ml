"""=============================================================================
Utility functions for creating neural network module.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

# ------------------------------------------------------------------------------

def sigmoid_prime(z):
    """Derivative of the sigmoid function. See:
    http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
    """
    return sigmoid(z) * (1 - sigmoid(z))

# ------------------------------------------------------------------------------

def init_biases(layers):
    """Assign random initial biases for each neuron in each layer using
    standard normal distribution.
    """
    # 1. Use `randn` because it uses the standard normal distribution.
    # 2. Each column vector of biases should be the same dimension as the
    #    number of nodes in the layer.
    # 3. Skip the first layer because it does not have biases.
    return [np.random.randn(n) for n in layers[1:]]

# ------------------------------------------------------------------------------

def init_weights(layers):
    """Assign random initial weights for between each set of LAYERS using the
    standard normal distribution.
    """
    weights = []
    for i, size in enumerate(layers):
        if i == len(layers)-1:
            break
        s1 = layers[i]
        s2 = layers[i + 1]
        # Use `randn` because it uses the standard normal distribution.
        W = np.random.randn(s1, s2)
        weights.append(W)
    return weights

# ------------------------------------------------------------------------------

def apply(list_a, list_b, fn):
    """Applies a function element-wise to ndarrays in two lists.
    """
    result = []
    for a, b in zip(list_a, list_b):
        assert a.shape == b.shape
        result.append(fn(a, b))
    return result

# ------------------------------------------------------------------------------

def shuffle(X, y):
    """Shuffles rows of matrix X and y in sync.
    """
    # Get randomly shuffled indices to ensure X and y are resorted
    # simultaneously.
    ix = np.arange(y.shape[0])
    np.random.shuffle(ix)
    X = X[ix, :]
    y = y[ix]
    return X, y


def quadratic_error(predictions, test_y):
    """Calculate error from quadratic cost function.
    """
    m = len(test_y)
    assert predictions.shape[0] == m
    error = 0
    for pred_arr, ans in zip(predictions, test_y):
        pred_arr[ans] = pred_arr[ans]-1
        norm = np.linalg.norm(pred_arr)
        error += norm**2
    return (1.0 / (2.0 * m)) * error

# ------------------------------------------------------------------------------

def y_vec_to_one_hot_arrays(y):
    """Converts a vector of y values into a matrix with one-hot arrays.
    """
    NUM_DIGITS = 10
    one_hots = np.zeros((y.size, NUM_DIGITS))
    for i, val in enumerate(y):
        one_hots[i] = y_val_to_one_hot_array(val)
    return one_hots

# ------------------------------------------------------------------------------

def y_val_to_one_hot_array(y):
    """Converts a value, e.g. 3, into a one-hot array
    """
    NUM_DIGITS = 10
    one_hot = np.zeros(NUM_DIGITS)
    one_hot[y] = 1.0
    return one_hot
