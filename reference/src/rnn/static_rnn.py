import tensorflow as tf
import numpy as np

n_inputs = 3    # number of features
n_neurons = 5   # number of neuron in single layer

X0 = tf.placeholder(tf.float32, [None, n_inputs])   # epoch size, n_features
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = output_seqs
init = tf.global_variables_initializer()

# mini batch        4 examples in the epoch
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1:X1_batch})
    print(Y0_val)  # output at time t= 0
    print(Y1_val)  # output at time t= 1
