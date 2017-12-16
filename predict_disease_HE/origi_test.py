import numpy as np 
import tensorflow as tf 
from PIL import Image 
import vgg19_new
import utils 
from tensorflow.python import debug as tf_dbg

img1 = utils.load_image('tiger.jpg')
img2 = utils.load_image('puzzle.jpg')

#Image.fromarray(img1.astype(np.uint8)).show()  
batch1 = img1.reshape(1, 224*224*3)
batch2 = img2.reshape(1, 224*224*3)

batch = np.concatenate((batch1, batch2),0)

#summary_op = tf.summary.merge_all()
rgb = tf.placeholder("float", [None, 224*224*3])
saver = tf.train.import_meta_graph('model/vgg19.ckpt.meta')
with tf.Session() as sess:
    #sess = tf_dbg.LocalCLIDebugWrapperSession(sess)     
    feed_dict = {rgb: batch}    
    #tf.global_variables_initializer().run() 
    saver.restore(sess, 'model/vgg19.ckpt')
    #writer = tf.summary.FileWriter('logs', tf.get_default_graph())    
    graph = tf.get_default_graph()
    print('\n'.join([n.name for n in graph.as_graph_def().node]))
    prob_op = graph.get_operation_by_name("content_vgg/prob")
    prob = sess.run(prob_op, feed_dict=feed_dict)
    #writer.add_graph(tf.get_default_graph())
    #print(prob.shape)
    #print(prob)
    utils.print_prob(prob[0], 'synset.txt')
    utils.print_prob(prob[1], 'synset.txt')



