from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

# Generate a set of data as the training set
x_train = np.float32(np.random.rand(100,2))
y_train = x_train * np.matrix([[0.2], [0.3]]) + 0.2 + np.random.normal(0, 0.01, 100).reshape(100, 1)

model = Sequential()
model.add(Dense(1, input_dim=2, kernel_initializer='RandomNormal', bias_initializer='RandomNormal'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='mse', optimizer=sgd)

# Train the model
model.fit(x_train, y_train, batch_size=1, verbose=1, epochs=200)

# Get the weight and bias from the trainable model
weight = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(weight, bias)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train[:,0], x_train[:,1], np.array(y_train), color = 'k')

xx, yy = np.meshgrid(np.arange(0.0, 1.0, 0.1), np.arange(0.0, 1.0, 0.1))
z = weight[0, 0] * xx + weight[1, 0] * yy + bias
ax.plot_surface(xx, yy, z, color = 'r')

plt.show()