import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request

from nltk.corpus import stopwords

os.chdir('/home/suri/PycharmProjects/Deep-Learning')
sess = tf.Session()

data_folder_name = 'data'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)


# model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size/2)
window_size = 3

# add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

# declare stop words
stops = stopwords.words('english')
# some test words. expecting synonymous to appear
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']

texts, target = text_helpers.load_movie_data(data_folder_name)


