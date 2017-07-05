import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate a set of data as the training set
x_data = np.float32(np.linspace(0, 1, 100))  # 100 random x data
x_data = x_data.reshape(1, 100)
y_data = 0.5 * x_data + 0.2 + np.random.normal(0, 0.01, 100)  # y_data = 0.5x + 0.2 + noise

# Create a TensorFlow linear model
b = tf.Variable(tf.zeros([1]))  # bias
W = tf.Variable(tf.random_uniform([1,1], -1.0, 1.0))  # weight
y = tf.matmul(W, x_data) + b  # y = W*x + b

# Settings of the solver
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Start training and print the parameters every 20 iterations
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# Draw the figure of the data and the result
y_estimated = sess.run(W) * x_data + sess.run(b)
x_data = x_data.reshape(100)
y_data = y_data.reshape(100)
y_estimated = y_estimated.reshape(100)
plt.axis([0,1,0,0.8])
plt.plot(x_data,y_data,'b.')
plt.plot(x_data, y_estimated,'r-')
plt.show()