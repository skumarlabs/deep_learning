from __future__ import print_function
import tensorflow as tf

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    return image        #a tensor


def input_pipeline(filenames, batchsize, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs,
                                                    shuffle=True)
    example, label = read_my_file_format(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batchsize
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], 
        batch_size = batchsize, 
        capacity=capacity, 
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

#create the graph
init_op = tf.global_variables_initializer()

#create a session
sess = tf.Session()

#initialize the variables (like the epoch counter)
sess.run(init_op)

pattern = "./../invasive_species_monitoring/test/*.jpg"
filenames = tf.train.match_filenames_once(pattern)

batch = input_pipeline(filenames, 50)


#start the input enqueue threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord = coord)



try:
    while not coord.should_stop():
        #run training steps or whatever
        image_tensor = sess.run([image])
        print(image_tensor)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    #When done, ask the thread to stop
    coord.request_stop()
coord.join(threads)
sess.close()
