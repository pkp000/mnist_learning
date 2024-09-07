import mnist_loader
import network
import numpy as np
import matplotlib.pyplot as plt

_, _, test_data = mnist_loader.load_data()
net = network.Network([784, 30, 10])

test_idx = 52

print('prediction:', np.argmax(net.feedforward(test_data[test_idx][0])[0]))
print('label:', test_data[test_idx][1])

fig, ax = plt.subplots()
ax.imshow(test_data[test_idx][0].reshape(28, 28), cmap='gray')
plt.show()