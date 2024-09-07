import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np


def load():
    # Read data with gzip and pickle.
    f = gzip.open('data\\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)


def load_data():
    tr_d, va_d, te_d = load()
    # please transform the format of tr_d, te_d to the following form:
    # list of 50000 training data, each containing a 2-tuple, where
    # tuple[0]: input image np_array.shape(1, 784)
    # tuple[1]: label np_array.shape(1, 10)
    # return (training_data, test_data)

    np_array.shape(1, 784)
    pass


def one_hot_enc(y):
    """Return a matrix of size (1, 10)
    The elements of the vector is 0 except for the yth position
    which is 1.0, indicating the digit of input.
    Example: y = 4, output: [[0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    """
    pass


if __name__ == '__main__':
    tr, ts = load_data()
    fig, axes = plt.subplots(3, 3)
    for row in range(3):
        for col in range(3):
            axes[row][col].imshow(tr[3 * row + col][0].reshape(28, 28), cmap='gray')
            axes[row][col].set_title(tr[3 * row + col][1])
    plt.show()