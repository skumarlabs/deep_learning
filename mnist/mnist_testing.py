import os
from datetime import datetime
from time import time

import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

os.chdir('../mnist')


def get_logdir_name():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    return logdir


BATCH_SIZE = 1000
NUM_EPOCHS = 1

print('reading input..')
t0 = time()
full_dataset = pd.read_csv('data/train.csv')
print('reading input done in: {:0.3f}s'.format(time() - t0))
full_features, full_labels = full_dataset.iloc[:, 1:], full_dataset.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(full_features, full_labels, test_size=0.10, random_state=5)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=5)
print('training set size', X_train.shape)
print('validation set size', X_valid.shape)
print('testing set size', X_test.shape)
pca = PCA(n_components=576)
print('running pca..')
t0 = time()
pca.fit(X_train)
print('pca done in: {:0.3f}s'.format(time() - t0))
X_train_pca = pca.transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

with tf.name_scope('input'):
    input_images = tf.constant(X_valid_pca)
    input_labels = tf.constant(y_valid)
# shuffles queue after each batch
image, label = tf.train.slice_input_producer([input_images, input_labels],
                                             num_epochs=NUM_EPOCHS)
images, labels = tf.train.batch([image, label],
                                batch_size=BATCH_SIZE,
                                allow_smaller_final_batch=True)
x = tf.placeholder(tf.float32, [None, 576])
y_ = tf.placeholder(tf.int64, [None])

with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 24, 24, 1])

with tf.name_scope('conv1'):
    W_conv1 = tf.get_variable(name='W_conv1', shape=[5, 5, 1, 32],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable(name='b_conv1', shape=[32], initializer=tf.constant_initializer(0.1))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv2'):
    W_conv2 = tf.get_variable(name='W_conv2', shape=[5, 5, 32, 64],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv2 = tf.get_variable(name='b_conv2', shape=[64], initializer=tf.constant_initializer(0.1))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

with tf.name_scope('pool2'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('fc1'):
    W_fc1 = tf.get_variable(name='W_fc1', shape=[6 * 6 * 64, 1024],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_fc1 = tf.get_variable(name='b_fc1', shape=[1024], initializer=tf.constant_initializer(0.1))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# map the 1024 features to 10 classes
with tf.name_scope('fc2'):
    W_fc2 = tf.get_variable(name='W_fc2', shape=[1024, 10], initializer=tf.truncated_normal_initializer())
    b_fc2 = tf.get_variable(name='b_fc2', shape=[10], initializer=tf.constant_initializer(0.1))
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('XE', cross_entropy)

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

config = tf.ConfigProto()
config.operation_timeout_in_ms = 2000000
summary_op = tf.summary.merge_all()

###########################################################################

with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver = tf.train.Saver()
    # saver.restore(sess, 'checkpoints/my_model_final.ckpt')
    saver.restore(sess, 'checkpoints/my_model-5000')
    print('Uninitialized variables: ', sess.run(tf.report_uninitialized_variables()))
    summary_writer = tf.summary.FileWriter(get_logdir_name())
    print('saving graph at', get_logdir_name())
    summary_writer.add_graph(sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    max_batch_num = len(X_train_pca) // BATCH_SIZE
    print('total examples: ', len(X_valid_pca))
    epoch_num = 0
    step = 1
    try:
        while not coord.should_stop():
            batch_x, batch_y = sess.run([images, labels])
            # images_ = images.eval()
            # print('batch shape', np.shape(batch_x), " and ", np.shape(batch_y))
            t0 = time()
            summary_str, loss_value, acc_score = sess.run([summary_op, cross_entropy, accuracy],
                                                          feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            duration = time() - t0
            print('step:{}, loss:{:.03f}, accuracy:{:.03f}, duration:{:.03f}s '.format(step, loss_value, acc_score,
                                                                                       duration))
            summary_writer.add_summary(summary_str, step)
            # recovered_images = pca.inverse_transform(images_)
            # Image.fromarray(recovered_images[0].reshape(28, 28)).resize((250, 250)).show()

            step += 1
    except tf.errors.OutOfRangeError:
        print('set finished at step:', step - 1)
    finally:
        coord.request_stop()
    coord.join(threads)
