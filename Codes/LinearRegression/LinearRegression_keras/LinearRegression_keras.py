from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt

x_train = np.float32(np.linspace(0, 1, 100))
y_train = 0.5 * x_train + 0.2 + np.random.normal(0, 0.01, 100)

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='zeros', bias_initializer='zeros'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='mse', optimizer=sgd)


model.fit(x_train, y_train, batch_size=1, epochs=200)

weight = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

plt.axis([0,1,0,0.8])
plt.plot(x_train,y_train,'b.')
plt.plot(x_train, weight * x_train + bias,'r-')
plt.show()