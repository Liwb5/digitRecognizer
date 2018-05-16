import dataProcess as dp
from model import MNIST

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import argparse
import sys


def main(_):
    
    train_images, train_labels, test_images = dp.loadData(args.trainPath, args.testPath)
    num_feature = train_images.shape[1]
    num_classes = train_labels.shape[1]
    
    mnist = MNIST(num_feature, num_classes)
    
    conv_L1 = mnist.addConvLayer(mnist.x_img,'conv_layer1',\
                            [3,3,1,32],[32],activation_func=tf.nn.relu,isDropout=True)
    conv_L2 = mnist.addConvLayer(conv_L1,'conv_layer2',\
                            [3,3,32,64],[64],activation_func=tf.nn.relu,isDropout=True)
    conv_L3 = mnist.addConvLayer(conv_L2, 'conv_layer3', \
                            [3,3,64,128], [128], activation_func=tf.nn.relu,isDropout=True)
    L3 = tf.reshape(conv_L3,[-1,4*4*128])

    fc_L1 = mnist.addFClayer(L3, 'fc_layer1',\
                            [4*4*128, 625],[625],activation_func=tf.nn.relu,isDropout=True)
    fc_L2 = mnist.addFClayer(fc_L1,'fc_layer2',[625, 10],[10])


    #prediction是softmax的结果，所以是概率值
    predict_probas = tf.nn.softmax(fc_L2)
    predicted_labels = tf.argmax(predict_probas, 1)#预测出类别

    #计算交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = mnist.input_labels,
                                                            logits = fc_L2)
    cost = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(args.lr).minimize(cross_entropy)

    # 预测对的个数，计算准确率
    correct_prediction = tf.equal(tf.argmax(predict_probas,1), tf.argmax(mnist.input_labels,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    saver = tf.train.Saver()#to save the model

    #通过tf.data这个类可以方便将数据
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    #batch 就是批量，repeat则可以无限循环，如果有参数，则是epoch的意思。
    dataset = dataset.batch(args.batch_size).repeat(args.maxEpoch)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()  #调用这个函数后面才可以直接使用eval函数。
    init = tf.global_variables_initializer()
    sess.run(init)    
    sess.run(iterator.initializer)#要注意初始化
    print_every = 5
    avg_cost = 0
    batch_count = 0
    while(1):
        try:
            batch_train, batch_label = sess.run(next_element)
            
        except tf.errors.OutOfRangeError:
            print('End of dataset')
            break
        else:
            feed_dict = {mnist.input_images: batch_train, 
                         mnist.input_labels: batch_label, 
                         mnist.keep_prob: args.drop_out }
            c, _ = sess.run([cost, train_step], feed_dict=feed_dict)
            avg_cost += c
            batch_count += 1
            if batch_count % print_every == 0:
                avg_cost = avg_cost / print_every
                print('batch_count:', '%04d' % (batch_count), 
                      'cost =', '{:.9f}'.format(avg_cost))
                avg_cost = 0
            
        
    #print('saving model ...')
    #saver.save(sess, '../models/model.tf')

    sess.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v:v.lower() == 'true')
    
    parser.add_argument(
        '--trainPath',
        type = str,
        help='Direcotory to load train data.')
    
    parser.add_argument(
        '--testPath',
        type = str,
        help='Direcotory to load test data.')
    
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.001,
        help = 'the learning rate of the model(default = 0.001)')
    
    parser.add_argument(
        '--maxEpoch',
        type = int,
        default = 20,
        help = 'the max number of epoch(default = 100)')
    
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 128,
        help = 'the batch size of train data(default = 128)')
    
    parser.add_argument(
        '--drop_out',
        type = float,
        default = 0.7,
        help = 'the probability of drop out(default = 0.5) ')

    
    args, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
    