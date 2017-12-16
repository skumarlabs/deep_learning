import numpy as np 
import tensorflow as tf 
import vgg_test
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import time
from datetime import datetime
import sys

CHANNEL_NUM = 3
OLD_IMAGE_SHAPE = [1024, 1024]
NEW_IMAGE_SHAPE = [224, 224]


data_root = '/home/suri/d/datasets/' 
test_data_root = os.path.join(data_root, 'test_')
test_desc_csv = os.path.join(data_root, 'test.csv')
test_desc = pd.read_csv(test_desc_csv)
test_rows = test_desc[['row_id', 'image_name']]
test_filepath_list = [os.path.join(test_data_root, filename) for filename in test_rows['image_name'].values]
test_row_ids = list(test_rows['row_id'].values)

TEST_SET_SIZE = len(test_filepath_list)
print("total test dataset size:", TEST_SET_SIZE) 
# ANN Architecture
n_outputs = 14
BATCH_SIZE = 1
with tf.device('/gpu:0'):
    def _read_my_file_format(filename_queue):
        reader = tf.WholeFileReader()
        image_name, image_file = reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)
        return image, image_name
    tf.reset_default_graph()

    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, channels=1)
        image_decoded.set_shape([None, None, None])        
        image_resized = tf.image.resize_images(image_decoded, NEW_IMAGE_SHAPE)
        image_resized.set_shape((NEW_IMAGE_SHAPE[0], NEW_IMAGE_SHAPE[1], 1)) 
        #flatten_image = tf.reshape(resized_img, [-1])
        rgb_img = tf.image.grayscale_to_rgb(image_resized, name="grey2rgb")
        scaled_img = tf.image.per_image_standardization(rgb_img)
        return scaled_img

    training = tf.placeholder(tf.int32, shape=[], name="is_training")
    
    test_images_tensor = tf.constant(test_filepath_list)       
    test_row_id_tensor = tf.constant(test_row_ids)    
    
    
    test_row_id, test_filename = tf.train.slice_input_producer([test_row_id_tensor, test_images_tensor], num_epochs=1, shuffle=False)
    test_scaled_img = _parse_function(test_filename)
    test_row_id_batch, test_image_batch = tf.train.batch([test_row_id, test_scaled_img], batch_size=BATCH_SIZE)   

    
    X = tf.placeholder(tf.float32, shape=([None , 224, 224, 3]), name="input_batch")
    y = tf.placeholder(tf.int64, shape=(None), name="labels")

    vgg = vgg_test.Vgg19()    
    with tf.name_scope("content_vgg"):
        vgg.build(X, training)
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=vgg.fc7)
        loss = tf.reduce_mean(xentropy, name="loss")        

    with tf.name_scope("eval"):
        correct =  tf.nn.in_top_k(vgg.fc7, y, 1) #tf.equal(tf.argmax(logits, 1), y) 
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        predictions = tf.argmax(vgg.fc7, 1)
    saver = tf.train.Saver()  
    with tf.Session() as sess:
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint('./model'))       
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        best_loss = 999999        
        result = "row_id,detected\n"
        batch_num = 0
        try:
            while not coord.should_stop():   
                id_batch, X_batch = sess.run([test_row_id_batch, test_image_batch], feed_dict={training:0})
                preds = sess.run([predictions],feed_dict={X:X_batch, training:0})
                for r_id, detected in zip(id_batch, preds[0]):
                   result += str(r_id.decode('utf-8')+ ",class_" + str(detected+1)+"\n")
                sys.stdout.write("images done# " + str(batch_num*len(id_batch)) + "\r")                
                batch_num+=1          
                
        except tf.errors.OutOfRangeError:
            with open('results.txt', 'w') as f:                
               f.write(result)
            coord.request_stop()
            coord.join(threads) 
        finally:    
            sess.close()

            

