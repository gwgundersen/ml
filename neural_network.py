"""============================================================================
Feedforward artificial neural network trained using backpropagation.
============================================================================"""

import numpy as np

# -----------------------------------------------------------------------------


class NeuralNetwork(object):

    def __init__(self, layers, biases=None, weights=None):
        """Initialize a neural network with optional biases and weights.
        """
        # Initialization code allows the user to set their own weights and
        # biases for testing purposes.
        self.biases = biases if biases else init_biases(layers)
        self.weights = weights if weights else init_weights(layers)
        self.layers = np.array(layers)

        assert len(self.biases) == len(self.layers) - 1
        assert len(self.weights) == len(self.layers) - 1

        # This check is actually critical. In `train`, we update the weights
        # and biases element-wise. If arrays in the weights and biases lists
        # are just Python lists, Python will concatenate the lists. We want
        # element-wise operations.
        assert [b_arr.dtype == np.float64 for b_arr in self.biases]

    def train(self, X, y, rate):
        """Updates neural network biases and weights after training on samples
        X and labels y with learning RATE eta.
        """
        # This ensures our `step` function divides correctly.
        assert type(rate) is float

        nabla_b_sum = [np.zeros(b.shape) for b in self.biases]
        nabla_W_sum = [np.zeros(W.shape) for W in self.weights]

        for x, y in zip(X, y):
            nabla_b, nabla_W = self._backpropagate(x, y)
            add = lambda a, b: a + b
            nabla_b_sum = apply(nabla_b_sum, nabla_b, add)
            nabla_W_sum = apply(nabla_W_sum, nabla_W, add)

        num_samples = X.shape[0]
        step = lambda a, b: a - (rate / num_samples) * b
        self.biases = apply(self.biases, nabla_b_sum, step)
        self.weights = apply(self.weights, nabla_W_sum, step)

    def train_SGD(self, X, y, rate, epochs, batch_size, test_X=None,
                  test_y=None):
        """Trains neural network using stochastic gradient descent with mini
        batches.
        """
        accuracy_per_epoch = []
        for epoch in range(epochs):
            # For each epoch:
            #   1. Shuffle data.
            #   2. Train on batches pulled sequentially from shuffled data.

            X, y = shuffle(X, y)

            num_samples = y.shape[0]
            assert num_samples % batch_size == 0
            # Cast to `int` because `range` cannot handle `np.float64` dtype.
            iterations = int(num_samples / batch_size)
            # Each iteration trains the neural network on a batch.
            print('total iterations: %s' % iterations)
            for i in range(iterations):
                print('iteration %s' % i)
                # For each mini-batch:
                #   1. Train on samples in mini-batch.
                #   2. Update weights and biases.
                start = i * batch_size
                end = start + batch_size
                X_batch = X[start:end, :]
                y_batch = y[start:end]
                self.train(X_batch, y_batch, rate)

            # If test files are given, evaluate error and accuracy of each
            # epoch.
            msg = '=====================\n' \
                  'Epoch\t%s\n' % epoch
            if hasattr(test_X, 'shape') and hasattr(test_y, 'shape'):
                accuracy, num_preds = self._accuracy(test_X, test_y)
                accuracy_per_epoch.append((accuracy, epoch))
                msg += 'Accuracy\t%s%%\n' % accuracy
                msg += '# Evictions\t%s\n' % np.round(num_preds, 2)
            print(msg)

        return accuracy_per_epoch if len(accuracy_per_epoch) > 0 else None

    def predict(self, X):
        """Predict labels for samples by feedforwarding data through network.
        """
        preds = []
        for x in X:
            activations, zeds = self._feedforward(x)
            preds.append(activations[-1])
        return np.array(preds)

    def _feedforward(self, a):
        """Perform feedforward on a single example.
        """
        activations = [a]
        # z_0 is irrelevant; insert dummy data to keep our lists aligned.
        zeds = [None]
        for j in range(len(self.layers)-1):
            W = self.weights[j]
            b = self.biases[j]
            z = np.dot(W.T, a) + b
            a = sigmoid(z)
            zeds.append(z)
            activations.append(a)
        return activations, zeds

    def _backpropagate(self, x, y):
        """Performs feedforward backpropagation against a single example and
        label.
        """
        activations, zeds = self._feedforward(x)

        # Compute the partial derivatives for the output layer.
        delta_b = (activations[-1] - y) * sigmoid_prime(zeds[-1])
        delta_W = np.outer(activations[-2], delta_b)
        nabla_b = [delta_b]
        nabla_W = [delta_W]

        # Compute the partial derivatives for any inner LAYERS, stepping back
        # from the second-to-last layer and stopping at the first layer.
        for j in range(len(self.layers)-2, 0, -1):
            W = self.weights[j]
            z = zeds[j]
            delta_b = np.dot(W, delta_b) * sigmoid_prime(z)
            a_upstream = activations[j-1]
            delta_W = np.outer(a_upstream, delta_b)
            # Insert at the beginning so that our nabla lists are properly
            # aligned with our weights and biases.
            nabla_b.insert(0, delta_b)
            nabla_W.insert(0, delta_W)

        return nabla_b, nabla_W

    def _accuracy(self, test_X, test_y):
        """Logs accuracy of current weights and biases using test data.
        """
        preds = self.predict(test_X)
        thresholded_preds = np.array([p.argmax() for p in preds])
        correct = thresholded_preds == test_y
        accuracy = (correct.sum() / float(test_y.size)) * 100
        return accuracy, thresholded_preds.sum()


# -----------------------------------------------------------------------------
# Utility functions.
# -----------------------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function. See:
    http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
    """
    return sigmoid(z) * (1 - sigmoid(z))


def init_biases(layers):
    """Assign random initial biases for each neuron in each layer using
    standard normal distribution.
    """
    # 1. Use `randn` because it uses the standard normal distribution.
    # 2. Each column vector of biases should be the same dimension as the
    #    number of nodes in the layer.
    # 3. Skip the first layer because it does not have biases.
    return [np.random.randn(n) for n in layers[1:]]


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


def apply(list_a, list_b, fn):
    """Applies a function element-wise to ndarrays in two lists.
    """
    result = []
    for a, b in zip(list_a, list_b):
        assert a.shape == b.shape
        result.append(fn(a, b))
    return result


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


def y_vec_to_one_hot_arrays(y):
    """Converts a vector of y values into a matrix with one-hot arrays.
    """
    NUM_DIGITS = 10
    one_hots = np.zeros((y.size, NUM_DIGITS))
    for i, val in enumerate(y):
        one_hots[i] = y_val_to_one_hot_array(val)
    return one_hots


def y_val_to_one_hot_array(y):
    """Converts a value, e.g. 3, into a one-hot array
    """
    NUM_DIGITS = 10
    one_hot = np.zeros(NUM_DIGITS)
    one_hot[y] = 1.0
    return one_hot
