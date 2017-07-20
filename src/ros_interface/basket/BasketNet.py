import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Truncated Image Range
x: 86-505
y: 56-265
"""

class BasketNet(object):
    def __init__(self):
        ckpt = 'ros_interface/basket/checkpoints/depthim.ckpt'
        self.input = tf.placeholder(tf.float32, [None, 209, 419, 1])
        self.var_dict = {}
        self.build()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=8,
            gpu_options=gpu_options))
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt)

    def build(self):
        print("Building net...")
        with tf.variable_scope('vgg_16'):
            conv1_1 = self.conv_layer(self.input, 1, 2, "conv1_1")

        with tf.variable_scope('fc'):
            fc6 = tf.nn.relu(self.fc_layer(conv1_1, 2*87571, 128, "fc6"))

            self.out8 = self.fc_layer(fc6, 128, 2, "out8")

        print("Net built.")

    def predict(self, ims):
        ims -= 1.5733289
        preds = self.sess.run(self.out8, feed_dict={self.input:ims})
        return preds

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.random_normal((filter_size, filter_size, in_channels, out_channels), stddev=1.4/np.sqrt(in_channels * filter_size**2)) # tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.zeros([out_channels]) # tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.random_normal((in_size, out_size), stddev=1.4/np.sqrt(in_size)) # tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.zeros([out_size]) # tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        value = initial_value
        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
