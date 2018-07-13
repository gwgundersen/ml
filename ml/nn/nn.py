"""=============================================================================
Feedforward artificial neural network trained using backpropagation.
============================================================================="""

import numpy as np
from   ml.nn import utils

# ------------------------------------------------------------------------------

class NeuralNetwork(object):

    def __init__(self, layers, biases=None, weights=None):
        """Initialize a neural network with optional biases and weights.
        """
        # Initialization code allows the user to set their own weights and
        # biases for testing purposes.
        self.biases = biases if biases else utils.init_biases(layers)
        self.weights = weights if weights else utils.init_weights(layers)
        self.layers = np.array(layers)

        assert len(self.biases) == len(self.layers) - 1
        assert len(self.weights) == len(self.layers) - 1

        # This check is actually critical. In `train`, we update the weights
        # and biases element-wise. If arrays in the weights and biases lists
        # are just Python lists, Python will concatenate the lists. We want
        # element-wise operations.
        assert [b_arr.dtype == np.float64 for b_arr in self.biases]

# ------------------------------------------------------------------------------

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
            nabla_b_sum = utils.apply(nabla_b_sum, nabla_b, add)
            nabla_W_sum = utils.apply(nabla_W_sum, nabla_W, add)

        num_samples = X.shape[0]
        step = lambda a, b: a - (rate / num_samples) * b
        self.biases = utils.apply(self.biases, nabla_b_sum, step)
        self.weights = utils.apply(self.weights, nabla_W_sum, step)

# ------------------------------------------------------------------------------

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

            X, y = utils.shuffle(X, y)

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

            # If test files are given, evaluate error and accuracy of each epoch.
            msg = '=====================\n' \
                  'Epoch\t%s\n' % epoch
            if hasattr(test_X, 'shape') and hasattr(test_y, 'shape'):
                accuracy, num_preds = self._accuracy(test_X, test_y)
                accuracy_per_epoch.append((accuracy, epoch))
                msg += 'Accuracy\t%s%%\n' % accuracy
                msg += '# Evictions\t%s\n' % np.round(num_preds, 2)
            print(msg)

        return accuracy_per_epoch if len(accuracy_per_epoch) > 0 else None

# ------------------------------------------------------------------------------

    def predict(self, X):
        """Predict labels for samples by feedforwarding data through network.
        """
        preds = []
        for x in X:
            activations, zeds = self._feedforward(x)
            preds.append(activations[-1])
        return np.array(preds)

# ------------------------------------------------------------------------------

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
            a = utils.sigmoid(z)
            zeds.append(z)
            activations.append(a)
        return activations, zeds

# ------------------------------------------------------------------------------

    def _backpropagate(self, x, y):
        """Performs feedforward backpropagation against a single example and
        label.
        """
        activations, zeds = self._feedforward(x)

        # Compute the partial derivatives for the output layer.
        delta_b = (activations[-1] - y) * utils.sigmoid_prime(zeds[-1])
        delta_W = np.outer(activations[-2], delta_b)
        nabla_b = [delta_b]
        nabla_W = [delta_W]

        # Compute the partial derivatives for any inner LAYERS, stepping back
        # from the second-to-last layer and stopping at the first layer.
        for j in range(len(self.layers)-2, 0, -1):
            W = self.weights[j]
            z = zeds[j]
            delta_b = np.dot(W, delta_b) * utils.sigmoid_prime(z)
            a_upstream = activations[j-1]
            delta_W = np.outer(a_upstream, delta_b)
            # Insert at the beginning so that our nabla lists are properly
            # aligned with our weights and biases.
            nabla_b.insert(0, delta_b)
            nabla_W.insert(0, delta_W)

        return nabla_b, nabla_W

# ------------------------------------------------------------------------------

    def _accuracy(self, test_X, test_y):
        """Logs accuracy of current weights and biases using test data.
        """
        preds = self.predict(test_X)
        thresholded_preds = np.array([p.argmax() for p in preds])
        correct = thresholded_preds == test_y
        accuracy = (correct.sum() / float(test_y.size)) * 100
        return accuracy, thresholded_preds.sum()
