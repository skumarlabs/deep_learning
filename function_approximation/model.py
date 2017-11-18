import tensorflow as tf


class Model:
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
        self.learning_rate = args.learning_rate
        input_dim = 1
        output_dim = 1
        hidden1_dim = 100

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, input_dim])
        self.targets = tf.placeholder(tf.float32, [args.batch_size, input_dim])  # for loss and optimization

        with tf.variable_scope("layer1"):
            W1 = tf.get_variable("layer1_w", shape=[input_dim, hidden1_dim], initializer=tf.random_normal_initializer)
            b1 = tf.get_variable("layer1_b", shape=[hidden1_dim], initializer=tf.random_normal_initializer)

        layer1 = tf.nn.relu(tf.add(tf.matmul(self.input_data, W1), b1))

        with tf.variable_scope("layer2"):
            W2 = tf.get_variable("layer2_w", shape=[hidden1_dim, output_dim], initializer=tf.random_normal_initializer)
            b2 = tf.get_variable("layer2_b", shape=[output_dim], initializer=tf.random_normal_initializer)

        with tf.variable_scope("logits"):
            self.pred = tf.add(tf.matmul(layer1, W2), b2)
            tf.summary.histogram('logits', self.pred)

        with tf.variable_scope('Loss'):
            self.loss = tf.reduce_mean(tf.square(self.pred - self.targets))
            # (Note the "_t" suffix here. It is pretty handy to avoid mixing
            # tensor summaries and th_eir actual computed summaries)
            tf.summary.scalar('scaler_loss', self.loss)
            tf.summary.histogram('train_loss', self.loss)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self, sess, x):
        feed = {self.input_data: [x]}
        [y_] = sess.run(self.pred, feed)
        return y_
