import mnist_loader
import network
import pickle

train_data, _, test_data = mnist_loader.load_data()
net = network.Network([784, 30, 10])
net.load_params('model_weights200.pkl', 'model_biases200.pkl')
net.SGD(train_data, 200, 10, 0.001, test_data=test_data)
net.save_params('model_weights400.pkl', 'model_biases400.pkl')