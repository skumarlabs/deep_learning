import os 
import tensorflow as tf 
import numpy as np 
import time 
import inspect 

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    
    def __init__(self, vgg19_npy_path=None):
        #pass 
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            
            path = os.path.join(path, 'data/vgg19.npy')
            vgg19_npy_path = path
            print('using npy file at', path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print('npy file loaded')
    
    def build(self, rgb, training):
        '''
        load variable from npy to build vgg
        param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        '''
        start_time = time.time()
        print('param build started')
        
        with tf.name_scope('reshape'):
            x_images_rgb = tf.reshape(rgb, [-1, 224, 224, 3])
            x_images_scaled_rgb = x_images_rgb #x_images_rgb * 255.0 
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x_images_scaled_rgb) 
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]

            # x_images_scaled_bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
            #  green - VGG_MEAN[1],
            #  red - VGG_MEAN[2]
            # ])
            x_images_bgr = tf.concat(axis=3, values=[blue, green, red])
        assert x_images_bgr.get_shape().as_list()[1:] == [224, 224, 3]
        ###################################   Block 1  ##################################################
        with tf.name_scope("conv1_1"):
            self.W_conv1_1 = self.get_conv_filter('conv1_1')
            self.conv1_1 = tf.nn.conv2d(x_images_bgr, self.W_conv1_1, [1, 1, 1, 1], padding="SAME")           
            self.b_conv1_1 = self.get_bias('conv1_1')
            self.relu1_1 = tf.nn.relu(tf.nn.bias_add(self.conv1_1, self.b_conv1_1)) 

        with tf.name_scope("conv1_2"):
            self.W_conv1_2 = self.get_conv_filter('conv1_2')
            self.conv1_2 = tf.nn.conv2d(self.relu1_1, self.W_conv1_2, [1, 1, 1, 1], padding="SAME")           
            self.b_conv1_2 = self.get_bias('conv1_2')
            self.relu1_2 = tf.nn.relu(tf.nn.bias_add(self.conv1_2, self.b_conv1_2))

        with tf.name_scope("pool1"):
            self.pool1 = self.max_pool_2x2(self.relu1_2, 'pool1')
            
        ##################################    Block 2   ##################################################
        with tf.name_scope("conv2_1"):
            self.W_conv2_1 = self.get_conv_filter('conv2_1')
            self.conv2_1 = tf.nn.conv2d(self.pool1, self.W_conv2_1, [1, 1, 1, 1], padding="SAME")           
            self.b_conv2_1 = self.get_bias('conv2_1')
            self.relu2_1 = tf.nn.relu(tf.nn.bias_add(self.conv2_1, self.b_conv2_1))

        with tf.name_scope("conv2_2"):
            self.W_conv2_2 = self.get_conv_filter('conv2_2')
            self.conv2_2 = tf.nn.conv2d(self.relu2_1, self.W_conv2_2, [1, 1, 1, 1], padding="SAME")           
            self.b_conv2_2 = self.get_bias('conv2_2')
            self.relu2_2 = tf.nn.relu(tf.nn.bias_add(self.conv2_2, self.b_conv2_2))  

        with tf.name_scope("pool2"):
            self.pool2 = self.max_pool_2x2(self.relu2_2, 'pool2')
        ##################################     Block 3   ##################################################
        with tf.name_scope("conv3_1"):
            self.W_conv3_1 = self.get_conv_filter('conv3_1')
            self.conv3_1 = tf.nn.conv2d(self.pool2, self.W_conv3_1, [1, 1, 1, 1], padding="SAME")           
            self.b_conv3_1 = self.get_bias('conv3_1')
            self.relu3_1 = tf.nn.relu(tf.nn.bias_add(self.conv3_1, self.b_conv3_1))

        with tf.name_scope("conv3_2"):
            self.W_conv3_2 = self.get_conv_filter('conv3_2')
            self.conv3_2 = tf.nn.conv2d(self.relu3_1, self.W_conv3_2, [1, 1, 1, 1], padding="SAME")           
            self.b_conv3_2 = self.get_bias('conv3_2')
            self.relu3_2 = tf.nn.relu(tf.nn.bias_add(self.conv3_2, self.b_conv3_2))  

        with tf.name_scope("conv3_3"):
            self.W_conv3_3 = self.get_conv_filter('conv3_3')
            self.conv3_3 = tf.nn.conv2d(self.relu3_2, self.W_conv3_3, [1, 1, 1, 1], padding="SAME")           
            self.b_conv3_3 = self.get_bias('conv3_3')
            self.relu3_3 = tf.nn.relu(tf.nn.bias_add(self.conv3_3, self.b_conv3_3))

        with tf.name_scope("conv3_4"):
            self.W_conv3_4 = self.get_conv_filter('conv3_4')
            self.conv3_4 = tf.nn.conv2d(self.relu3_3, self.W_conv3_4, [1, 1, 1, 1], padding="SAME")           
            self.b_conv3_4 = self.get_bias('conv3_4')
            self.relu3_4 = tf.nn.relu(tf.nn.bias_add(self.conv3_4, self.b_conv3_4))

        with tf.name_scope("pool3"):
            self.pool3 = self.max_pool_2x2(self.relu3_4, 'pool3')
            self.dropout3 = tf.cond(training>0, lambda: tf.nn.dropout(self.pool3, keep_prob=0.5), lambda:self.pool3)
        #conv3_stop = tf.stop_gradient(self.pool3)
        #################################   Block 4    ################################################
        with tf.name_scope("conv4_1"):
            self.W_conv4_1 = self.get_conv_filter('conv4_1')
            self.conv4_1 = tf.nn.conv2d(self.dropout3, self.W_conv4_1, [1, 1, 1, 1], padding="SAME")           
            self.b_conv4_1 = self.get_bias('conv4_1')
            self.relu4_1 = tf.nn.relu(tf.nn.bias_add(self.conv4_1, self.b_conv4_1))

        with tf.name_scope("conv4_2"):
            self.W_conv4_2 = self.get_conv_filter('conv4_2')
            self.conv4_2 = tf.nn.conv2d(self.relu4_1, self.W_conv4_2, [1, 1, 1, 1], padding="SAME")           
            self.b_conv4_2 = self.get_bias('conv4_2')
            self.relu4_2 = tf.nn.relu(tf.nn.bias_add(self.conv4_2, self.b_conv4_2))  

        with tf.name_scope("conv4_3"):
            self.W_conv4_3 = self.get_conv_filter('conv4_3')
            self.conv4_3 = tf.nn.conv2d(self.relu4_2, self.W_conv4_3, [1, 1, 1, 1], padding="SAME")           
            self.b_conv4_3 = self.get_bias('conv4_3')
            self.relu4_3 = tf.nn.relu(tf.nn.bias_add(self.conv4_3, self.b_conv4_3))

        with tf.name_scope("conv4_4"):
            self.W_conv4_4 = self.get_conv_filter('conv4_4')
            self.conv4_4 = tf.nn.conv2d(self.relu4_3, self.W_conv4_4, [1, 1, 1, 1], padding="SAME")           
            self.b_conv4_4 = self.get_bias('conv4_4')
            self.relu4_4 = tf.nn.relu(tf.nn.bias_add(self.conv4_4, self.b_conv4_4))

        with tf.name_scope("pool4"):
            self.pool4 = self.max_pool_2x2(self.relu4_4, 'pool4')
            self.dropout4 = tf.cond(training>0, lambda: tf.nn.dropout(self.pool4, keep_prob=0.5), lambda:self.pool4)
        
        ###################################   Block 5   #############################################
        with tf.name_scope("conv5_1"):
            self.W_conv5_1 = self.get_conv_filter('conv5_1')
            self.conv5_1 = tf.nn.conv2d(self.dropout4, self.W_conv5_1, [1, 1, 1, 1], padding="SAME")           
            self.b_conv5_1 = self.get_bias('conv5_1')
            self.relu5_1 = tf.nn.relu(tf.nn.bias_add(self.conv5_1, self.b_conv5_1))
            
        with tf.name_scope("conv5_2"):
            self.W_conv5_2 = self.get_conv_filter('conv5_2')
            self.conv5_2 = tf.nn.conv2d(self.relu5_1, self.W_conv5_2, [1, 1, 1, 1], padding="SAME")           
            self.b_conv5_2 = self.get_bias('conv5_2')
            self.relu5_2 = tf.nn.relu(tf.nn.bias_add(self.conv5_2, self.b_conv5_2))  

        with tf.name_scope("conv5_3"):
            self.W_conv5_3 = self.get_conv_filter('conv5_3')
            self.conv5_3 = tf.nn.conv2d(self.relu5_2, self.W_conv5_3, [1, 1, 1, 1], padding="SAME")           
            self.b_conv5_3 = self.get_bias('conv5_3')
            self.relu5_3 = tf.nn.relu(tf.nn.bias_add(self.conv5_3, self.b_conv5_3))

        with tf.name_scope("conv5_4"):
            self.W_conv5_4 = self.get_conv_filter('conv5_4')
            self.conv5_4 = tf.nn.conv2d(self.relu5_3, self.W_conv5_4, [1, 1, 1, 1], padding="SAME")           
            self.b_conv5_4 = self.get_bias('conv5_4')
            self.relu5_4 = tf.nn.relu(tf.nn.bias_add(self.conv5_4, self.b_conv5_4))

        with tf.name_scope("pool5"):
            self.pool5 = self.max_pool_2x2(self.relu5_4, 'pool5')
            self.dropout5 = tf.cond(training>0, lambda: tf.nn.dropout(self.pool5, keep_prob=0.5), lambda:self.pool5)
        
        ##################################   FC6 Layer   #############################################

        with tf.name_scope('fc6'):
            shape = self.dropout5.get_shape().as_list()
            dim = 1 
            for d in shape[1:]:     # flattened size
                dim *= d  
            pool5_flat = tf.reshape(self.dropout5, [-1, dim])

            self.W_fc6 = tf.get_variable(shape=[dim, 512], name='W_fc6') #self.get_fc_weight('fc6') #
            self.b_fc6 = tf.get_variable(shape=[512], name='b_fc6') #self.get_bias('fc6') 
            self.fc6 = tf.nn.bias_add(tf.matmul(pool5_flat, self.W_fc6), self.b_fc6)
            assert self.fc6.get_shape().as_list()[1:] == [512]
            self.relu6 = tf.nn.relu(self.fc6)
        ###################################   FC7 Layer   #############################################
        with tf.name_scope('fc7'):
            shape = self.relu6.get_shape().as_list()
            dim = 1 
            for d in shape[1:]:     # flattened size
                dim *= d  
            relu6_flat = tf.reshape(self.relu6, [-1, dim])

            self.W_fc7 = tf.get_variable(shape=[dim, 14], name='W_fc7') #self.get_fc_weight('fc7')
            self.b_fc7 = tf.get_variable(shape=[14], name='b_fc7') #self.get_bias('b_fc7')
            self.fc7 = tf.nn.bias_add(tf.matmul(relu6_flat, self.W_fc7), self.b_fc7)
            assert self.fc7.get_shape().as_list()[1:] == [14]
            #self.relu7 = tf.nn.relu(self.fc7)
        ###################################   FC8 Layer   ############################################
        # with tf.name_scope('fc8'):
        #     shape = self.relu7.get_shape().as_list()
        #     dim = 1 
        #     for d in shape[1:]:     # flattened size
        #         dim *= d  
            
        #     relu7_flat = tf.reshape(self.relu7, [-1, dim])

        #     self.W_fc8 = self.get_fc_weight('fc8')
        #     self.b_fc8 = self.get_bias('fc8')
        #     self.fc8 = tf.nn.bias_add(tf.matmul(relu7_flat, self.W_fc8), self.b_fc8)           
        ###################################   Softmax Layer   #########################################
        self.prob = tf.nn.softmax(self.fc7, name="prob")
        self.data_dict = None 
        print(("build model finished: %ds" %(time.time()-start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_2x2(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_conv_filter(self, name):
        intial = tf.constant(self.data_dict[name][0], name='filter')
        return tf.get_variable(initializer=intial, name="W_"+name)

    def get_bias(self, name):
        initial = tf.constant(self.data_dict[name][1], name='biases')
        return tf.get_variable(initializer=initial, name="b_"+name)

    def get_fc_weight(self, name):
        initial = tf.constant(self.data_dict[name][0], name='weights')
        return tf.get_variable(initializer=initial, name="W_"+name)










        




