import dataProcess as dp
from model import MNIST

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf


#set max number of epoch
MAX_NUM_EPOCH = 20

# settings
LEARNING_RATE = 0.001

DROPOUT = 0.7
BATCH_SIZE = 100


trainPath = '../data/train.csv'
testPath = '../data/test.csv'
train_images, train_labels, test_images = dp.loadData(trainPath, testPath)

num_feature = train_images.shape[1]
num_classes = train_labels.shape[1]

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


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
# prediction = mnist.get_prob(fc_L1, [1024, 10],[10])

#prediction是softmax的结果，所以是概率值
prediction = tf.nn.softmax(fc_L2)
predicted_labels = tf.argmax(prediction, 1)#预测出类别

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = mnist.input_labels,logits = fc_L2)
cost = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


# 预测对的个数
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(mnist.input_labels,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

saver = tf.train.Saver()#to save the model


sess = tf.InteractiveSession()  #调用这个函数后面才可以直接使用eval函数。
init = tf.global_variables_initializer()
sess.run(init)
print_every = 5
print_cost = 0
for epoch in range(MAX_NUM_EPOCH):
    avg_cost = 0
    total_batch = int(train_images.shape[0]/BATCH_SIZE)
    for i in range(total_batch):

        batch_xs, batch_ys = next_batch(BATCH_SIZE)
        feed_dict = {mnist.input_images: batch_xs, mnist.input_labels: batch_ys, mnist.keep_prob:DROPOUT }
        c, _ = sess.run([cost, train_step], feed_dict=feed_dict)
        avg_cost += c / total_batch
        print_cost += c
        if (i+1)%print_every == 0:
            print_cost = print_cost / print_every
            print('Epoch:', '%04d' % (epoch + 1), '| cost =', '{:.9f}'.format(print_cost))
            print_cost = 0

#print('saving model ...')
#saver.save(sess, '../models/model.tf')

sess.close()
