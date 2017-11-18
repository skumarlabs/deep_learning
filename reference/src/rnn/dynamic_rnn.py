import tensorflow as tf
import numpy as np

n_inputs = 3    # number of features
n_neurons = 5   # number of neuron in single layer
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

# mini batch        4 examples in the epoch
X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])
with tf.Session() as sess:
    init.run()
    outputs_val = sess.run([outputs], feed_dict={X: X_batch})
    print(outputs_val)  # outputs

