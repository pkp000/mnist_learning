import random
import numpy as np
import pickle


class Network:
    def __init__(self, sizes):
        """
        :param sizes: list of neuron numbers in each layer, e.g. [784, 15, 10]
        """
        self.nb_layers = len(sizes)  # XXXXXX Save neural network layer number
        self.sizes = sizes  # Save neuron numbers of each layer

        # XXX List of numpy arrays, save bias parameters of each layer,
        # from layer 1 ~ (n-1), 1st layer excluded.
        # e.g. [np_array1:shape = (1,15), np_array2:shape = (1, 10)]
        # Initialize to random values: np.random.randn
        self.biases = [np.random.rand(1, s) for s in sizes[1:]]
        # XXX e.g. [np_array1:shape = (784,15), np_array2(shape = (1, 10)]
        # Initialize to random values: np.random.randn
        self.weights = [np.random.rand(j, k)
                        for j, k in zip(sizes[0:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network for input 'a'.
        :param a: input to the network, e.g. np_array:shape(1, 784)
        :return: network output, e.g. shape(1, 10)
        """
        # XXX loop through bias&weights parameters, and calculate a @ w + b.
        return a

    def SGD(self, train_data, epochs, mini_batch_size, learning_rate,
            test_data):
        """Train the neural network using mini-batch stochastic gradient
        descent.
        :param train_data: list of tuples, length 50000.
        tuple[0]: vectorized image np_array:shape(1, 784)
        tuple[1]: one-hot encoded label np_array:shape(1, 10)
        :param epochs: number of epochs to train.
        :param test_data: list of tuples, length 10000.
        """
        n = len(train_data)
        for j in range(epochs):
            # XXX shuffle training data at the beginning of every epoch
            # random.shuffle

            # XXX construct the list of mini_batches with each mini_batch
            # containing mini_batch_size training data.
            mini_batches = None

            # update bias and weights parameters with each mini_batch
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, learning_rate)

            # print test result, e.g. Epoch_0: 100/10000
            print('Epoch_{}: {} / {}'.format(
                j, self.evaluate(test_data), len(test_data)))

    def gradient_descent(self, mini_batch, learning_rate):
        """Update the networks weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        :param mini_batch: list of tuples [(x1, y1),...]
        """
        # partial_b_sum sums the partial_b of each training data of
        # the mini_batch.
        partial_b_sum = [np.zeros(b.shape) for b in self.biases]
        # partial_w_sum sums the partial_w of each training data of
        # the mini_batch.
        partial_w_sum = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # XXX call back propagation for each training data and
            # update partial_w_sum, partial_b_sum.
            pass

        # XXX Update biases and weights according to gradient descent formula
        # self.biases
        # self.weights

    def backprop(self, x, y):
        """Return a tuple of (,) representing the
        gradient for the cost function C_x.
        """

        # partial_b, partial_w stores the the partial derivatives of
        # cost function C with regard to each parameter b and w for a
        # single input. Initialized to zero.
        partial_b = [np.zeros(b.shape) for b in self.biases]
        partial_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward

        # Activation of the first layer == input x.
        activation = x

        # List which stores the activation vector of each layer.
        activations = [x]

        # List which stores the z vector of each layer.
        zs = []  # List to store z vector of each layer.

        # XXX populate activations, and zs lists in one forward pass.
        for b, w in zip(self.biases, self.weights):
            pass

        # backward

        # XXX calculate delta value of the last layer, formula BP1
        # delta =

        # XXX calculate partial_C/partial_b of the last layer BP3
        # partial_b[-1]

        # XXX calculate partial_C/partial_w of the last layer BP4
        # partial_w[-1]

        # back propagate delta value
        # calculate delta[-2], partial_b[-2], partial_w[-2] until [-(nb_layers-1)](layer 2)
        for l in range(2, self.nb_layers):
            # XXX calculate delta of the layer[-l]. BP2
            # delta =

            # XXX calculate partial_C/partial_b of the layer[-l] BP3
            # partial_b[-l] =

            # XXX calculate partial_C/partial_w of the last layer BP4
            # partial_w[-l]

            return partial_b, partial_w


    def cost_derivative(self, output_activation, y):
        """return the vector of partial derivatives
        partial(C_x)/partial(a)
        :param output_activation: output vector of shape (1, n)
        :param y: training data label in one-hot coding of shape (1, n)
        """
        # XXX
        return None

    def evaluate(self, test_data):
        """Return the number of test inputs for which the
        neural network gives correct predictions.
        :param test_data: iterator object, return tuple on each iteration,
        (input_x - shape [784,1], label_x - shape [10, 1])
        """
        # XXX calculate the total number of correct prediction under the
        # current network parameters (w,b).
        # return

    def save_params(self, weights, biases):
        with open(weights, 'wb') as f:
            pickle.dump(self.weights, f)
        with open(biases, 'wb') as f:
            pickle.dump(self.biases, f)

    def load_params(self, weights, biases):
        with open(weights, 'rb') as f:
            self.weights = pickle.load(f)
        with open(biases, 'rb') as f:
            self.biases = pickle.load(f)

    def __str__(self):
        return str(self.sizes)


def sigmoid(z):
    """
    XXX
    :param z: numpy array (supporting element wise operations)
    :return: numpy array of sigmoid of each element of z.
    """
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    XXX
    Derivative of sigmoid function.
    :param z: numpy array (supporting element-wise operations)
    """
    return None


if __name__ == '__main__':
    net = Network([2, 15, 10])
    a = np.array()
    print(net.feedforward(a))
