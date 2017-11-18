import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
logdir = "{}/run-{}/".format(root_logdir, now)

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01
tf.reset_default_graph()

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
W = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="W")
y_pred = tf.matmul(X, W, name="predictions")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="loss/mse") # get name by mse.op.name
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse, name="training_op")


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # sess.run(init)
    # to restore instead of tf.global_variables_initializer(),
    # call saver.restore() and call tf.reset_default_graph() if needed.

    # to restore with graph definition( imports graph into default graph) call tf.train.import_meta_graph().restore()
    # saver = tf.train.import_meta_graph("/tmp/model/my_model_final.ckpt.meta")
    saver.restore(sess, "/tmp/model/my_model_final.ckpt")
    graph = tf.get_default_graph()
    # W = graph.get_tensor_by_name("W:0")
    # mse = graph.get_tensor_by_name("mse:0")
    # training_op = graph.get_operation_by_name("training_op:0")

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for epoch in range(n_epochs):
        if epoch % 100 == 0:  # checkpoint every 100 epoch
            print("Epoch", epoch, "MSE =", mse.eval())
            summary_str = mse_summary.eval()
            file_writer.add_summary(summary_str)
            save_path = saver.save(sess, "/tmp/model/my_model.ckpt")

        sess.run(training_op)

    best_theta = W.eval()
    best_save_path = saver.save(sess, "/tmp/model/my_model_final.ckpt")
    file_writer.close()
