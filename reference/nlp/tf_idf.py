import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np

import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir('/home/suri/PycharmProjects/Deep-Learning')
sess = tf.Session()
batch_size = 200
max_features = 1000

save_file_name = os.path.join('data/smsspamcollection', 'SMSSpamCollection')
text_data = []
temp_output_file = open(save_file_name, 'r')
file = temp_output_file.read()
# reformat dataset
text_data = file.encode('ascii', errors='ignore')  # returns bytes repr of string
text_data = text_data.decode()  # returns str in UTF-8
text_data = text_data.split('\n')
text_data = [x.split('\t') for x in text_data if len(x) >= 1]

texts = [x[1] for x in text_data]
targets = [x[0] for x in text_data]
targets = [1 if x == 'spam' else 0 for x in targets]

texts = [text.lower() for text in texts]  # convert to lowercase
texts = [''.join(char for char in text if char not in string.punctuation) for text in texts]  # remove punctuation
texts = [''.join(char for char in text if char not in '0123456789') for text in texts]  # remove numbers
texts = [' '.join(text.split()) for text in texts]


def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


# Create tf-idf of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8 * sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))

texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]

target_train = np.array([label for index, label in enumerate(targets) if index in train_indices])
target_test = np.array([label for index, label in enumerate(targets) if index in test_indices])

A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []

for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
        if (i + 1) % 500 == 0:
            acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
            acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
            print(
                'Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
                    *acc_and_loss))
