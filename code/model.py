# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

class MNIST(object):
    def __init__(self,image_size, num_classes):
        self.input_images = tf.placeholder(tf.float32, [None, image_size], name = 'input_images')
        width = height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
        self.x_img = tf.reshape(self.input_images, [-1,width,height,1])
        self.input_labels = tf.placeholder(tf.float32, [None, num_classes], name = 'input_labels')
        self.keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')

    def addConvLayer(self,inputs,name_scope,weight_shape, bias_shape, activation_func = None, isDropout=False):
        with tf.name_scope(name_scope):
            weight = self.weight_variable(shape = weight_shape)
            bias = self.bias_variable(shape = bias_shape)
            con2d = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')
            L_output = activation_func(con2d+bias)
            L_pool = tf.nn.max_pool(L_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if isDropout:
                L_pool = tf.nn.dropout(L_pool, keep_prob=self.keep_prob)
            return L_pool

    def addFClayer(self,inputs, name_scope, weight_shape, bias_shape,activation_func=None,isDropout=False):
        with tf.name_scope(name_scope):
            w_fc = self.weight_variable(weight_shape)
            b_fc = self.bias_variable(bias_shape)
            h_fc = activation_func(tf.matmul(inputs, w_fc)+b_fc)
            if isDropout:
                h_fc = tf.nn.dropout(h_fc, keep_prob=self.keep_prob)
            return h_fc

    def get_prob(self, inputs, weight_shape, bias_shape):
        w = self.weight_variable(weight_shape)
        b = self.bias_variable(bias_shape)
        return tf.nn.softmax(tf.matmul(inputs, w)+b)

    def get_loss(self,pred):
        # the error between prediction and real data
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.input_labels * tf.log(pred),
                                                      reduction_indices=[1]))       # loss
        return cross_entropy

    # weight initialization
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
