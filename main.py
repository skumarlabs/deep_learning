import numpy as np 
import tensorflow as tf 
from PIL import Image 
import vgg19_new
import utils 
from tensorflow.python import debug as tf_dbg 
import pandas as pd
import PIL
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import json
import glob
from functools import partial
import time
from datetime import datetime

IMG_DIM = (224 , 224)
CHANNEL_NUM = 3
OLD_IMAGE_SHAPE = [1024, 1024]
NEW_IMAGE_SHAPE = [224, 224]


data_root = '/home/suri/d/datasets/' 
train_data_root = os.path.join(data_root, 'train_')
test_data_root = os.path.join(data_root, 'test_')
os.listdir(data_root)
class_maps_json = os.path.join(data_root, 'class_maps.json')
train_desc_csv = os.path.join(data_root, 'train.csv')
test_desc_csv = os.path.join(data_root, 'test.csv')
sample_submission_csv = os.path.join(data_root, 'sample_submission_csv')
with open(class_maps_json) as f:
    class_maps = json.load(f)

train_desc = pd.read_csv(train_desc_csv)
test_desc = pd.read_csv(test_desc_csv)

######################################################
dev_labels = train_desc[['image_name', 'detected']]
dev_labels_encoded = [int(i[6:])-1 for i in dev_labels['detected']]
dev_labels = dev_labels.assign(target=dev_labels_encoded)
dev_labels.drop(['detected'], axis=1, inplace=True)
#######################################################
filename_pattern  = os.path.join(data_root,'train_/*.png')
filepath_list = [os.path.join(train_data_root, filename) for filename in dev_labels['image_name'].values]
print("total dev dataset size:", len(filepath_list))
labels = list(dev_labels['target'].values)
#########################################################
NUM_CLASSES = max(dev_labels['target'].unique())
print("Total unique classes:", NUM_CLASSES)

class_0 = dev_labels[dev_labels['target'] == 0]
class_1 = dev_labels[dev_labels['target'] == 1]
class_2 = dev_labels[dev_labels['target'] == 2]
class_3 = dev_labels[dev_labels['target'] == 3]
class_4 = dev_labels[dev_labels['target'] == 4]
class_5 = dev_labels[dev_labels['target'] == 5]
class_6 = dev_labels[dev_labels['target'] == 6]
class_7 = dev_labels[dev_labels['target'] == 7]
class_8 = dev_labels[dev_labels['target'] == 8]
class_9 = dev_labels[dev_labels['target'] == 9]
class_10 = dev_labels[dev_labels['target'] == 10]
class_11 = dev_labels[dev_labels['target'] == 11]
class_12 = dev_labels[dev_labels['target'] == 12]
class_13 = dev_labels[dev_labels['target'] == 13]

print("before resampling, values counts", dev_labels['target'].value_counts().sort_index())

majority_class_size = class_6.shape[0]
print("majority class size:", majority_class_size)
print("resampling other classes..")

class_0_resampled = resample(class_0, replace=True, n_samples=majority_class_size, random_state=5)
class_1_resampled = resample(class_1, replace=True, n_samples=majority_class_size, random_state=5)
class_2_resampled = resample(class_2, replace=False, n_samples=majority_class_size, random_state=5)
class_3_resampled = resample(class_3, replace=True, n_samples=majority_class_size, random_state=5)
class_4_resampled = resample(class_4, replace=True, n_samples=majority_class_size, random_state=5)
class_5_resampled = resample(class_5, replace=True, n_samples=majority_class_size, random_state=5)
class_7_resampled = resample(class_7, replace=True, n_samples=majority_class_size, random_state=5)
class_8_resampled = resample(class_8, replace=True, n_samples=majority_class_size, random_state=5)
class_9_resampled = resample(class_9, replace=True, n_samples=majority_class_size, random_state=5)
class_10_resampled = resample(class_10, replace=True, n_samples=majority_class_size, random_state=5)
class_11_resampled = resample(class_11, replace=True, n_samples=majority_class_size, random_state=5)
class_12_resampled = resample(class_12, replace=True, n_samples=majority_class_size, random_state=5)
class_13_resampled = resample(class_13, replace=True, n_samples=majority_class_size, random_state=5)

df_upsampled = pd.concat([class_6,
                        class_0_resampled,
                        class_1_resampled,
                        class_2_resampled,
                        class_3_resampled,
                        class_4_resampled,
                        class_5_resampled,
                        class_7_resampled,
                        class_8_resampled,
                        class_9_resampled,
                        class_10_resampled,
                        class_11_resampled,
                        class_12_resampled,
                        class_13_resampled])

filepath_list = [os.path.join(train_data_root, filename) for filename in df_upsampled['image_name'].values]
labels = list(df_upsampled['target'].values)

print("After resampling, values counts", df_upsampled['target'].value_counts().sort_index())

image_files_train, image_files_test, labels_train, labels_test = train_test_split(filepath_list, labels, stratify=labels, test_size=0.15, random_state=5)
TRAIN_SET_SIZE = len(image_files_train)
print("Total dev dataset size:", len(filepath_list))
print("Total train dataset size:", TRAIN_SET_SIZE)




def _read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    image_name, image_file = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    return image, image_name



 # define pattern

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
logdir = "{}/run-{}".format(root_logdir, now)

# ANN Architecture
n_outputs = NUM_CLASSES+1
VGG_MEAN = [103.939, 116.779, 123.68]
BATCH_SIZE = 25
learning_rate = 0.0001



tf.reset_default_graph()



with tf.device('/cpu:0'):
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, channels=1)
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, NEW_IMAGE_SHAPE)
        return image_resized
    filenames_tensor = tf.constant(image_files_train) 
    labels_tensor = tf.constant(labels_train)
    filename, label = tf.train.slice_input_producer([filenames_tensor, labels_tensor], shuffle=True, num_epochs=10, capacity=TRAIN_SET_SIZE) # A list of tensors, one for each element of tensor_list
    #print(filename)
    resized_img = _parse_function(filename)
    resized_img.set_shape((NEW_IMAGE_SHAPE[0], NEW_IMAGE_SHAPE[1], 1)) 
    flatten_image = tf.reshape(resized_img, [-1])
    rgb_img = tf.image.grayscale_to_rgb(resized_img, name="grey2rgb")
    image_batch, label_batch = tf.train.batch([rgb_img, label], batch_size=BATCH_SIZE, allow_smaller_final_batch=False)   
    global_step = tf.Variable(1, name='global_step',trainable=False)
    X = tf.placeholder(tf.float32, shape=([None , 224, 224, 3]), name="input_batch")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

vgg = vgg19_new.Vgg19()    
with tf.name_scope("content_vgg"):
    vgg.build(X)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=vgg.fc7)
    loss = tf.reduce_mean(xentropy, name="loss")
    


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)


with tf.name_scope("eval"):
    correct =  tf.nn.in_top_k(vgg.fc7, y, 1) #tf.equal(tf.argmax(logits, 1), y) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("summary"):
    loss_summary = tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir)


#(Image.fromarray(X_batch[0].reshape(256, 256)).show())
#print(y_batch[0], X_batch[0].reshape(-1, 1).shape)

#summary_op = tf.summary.merge_all() 
saver = tf.train.Saver()      
saver2 = tf.train.Saver()

NUM_BATCH = TRAIN_SET_SIZE/BATCH_SIZE
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.log_device_placement=True
config.operation_timeout_in_ms = 2000000

with tf.Session(config=config) as sess:
    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    #sess = tf_dbg.LocalCLIDebugWrapperSession(sess)
    #saver.restore(sess, 'model/vgg19.ckpt')
    #init = tf.variables_initializer(
    #                            [v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))]
    #                        )
    sess.run(init)
    
    writer.add_graph(tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop():
            X_batch, y_batch = sess.run([image_batch, label_batch])
                        
            glob_step = tf.train.global_step(sess, global_step)

            if glob_step % 10 == 0:
                #pass
                
                _,summary_str, acc, loss_ = sess.run([training_op, loss_summary, accuracy, loss],feed_dict={X:X_batch, y:y_batch})
                writer.add_summary(summary_str, global_step=glob_step)
                print("step:", glob_step, "acc:", acc, "loss:", loss_)
                #print("time taken in the batch", time.time()-t0, 's') 
                save_path = saver2.save(sess, "./model/my_model.ckpt", global_step=glob_step) # checkpoint every 100 step
            else:
                t0 = time.time()
                #print(X_batch.shape)
                #(Image.fromarray(X_batch[0].reshape(224, 224, 3).astype(np.uint8)).show())
                _ = sess.run([training_op],feed_dict={X:X_batch, y:y_batch})
                print("batch time {:.4}s, step# {}".format(time.time()-t0, glob_step))
                
                
                
                
    except tf.errors.OutOfRangeError:
        print("model saved at step=", tf.train.global_step(sess, glob_step))
        save_path = saver2.save(sess, "./model/my_model_final.ckpt", global_step=glob_step)    
        file_writer.close()
    finally:
        coord.request_stop()
        coord.join(threads)     
        sess.close()

        

