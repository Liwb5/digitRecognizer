import dataProcess as dp
from model import MNIST

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf


# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 0

# image number to output
IMAGE_TO_DISPLAY = 10


trainPath = '../data/train.csv'
testPath = '../data/test.csv'
train_images, train_labels, test_images = dp.loadData(trainPath, testPath)
num_feature = train_images.shape[1]
print(num_feature)

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




mnist = MNIST(784, 10)
conv_L1 = mnist.addConvLayer(mnist.x_img,'conv_layer1',[5,5,1,32],[32],activation_func=tf.nn.relu,isDropout=True)
conv_L2 = mnist.addConvLayer(conv_L1,'conv_layer2',[5,5,32,64],[64],activation_func=tf.nn.relu,isDropout=True)
L2 = tf.reshape(conv_L2,[-1,7*7*64])
fc_L1 = mnist.addFClayer(L2, 'fc_layer1',[7*7*64, 1024],[1024],activation_func=tf.nn.relu,isDropout=True)
prediction = mnist.get_prob(fc_L1, [1024, 10],[10])

cross_entropy = mnist.get_loss(prediction)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(mnist.input_labels,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1
for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    # if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
    #
    #     train_accuracy = accuracy.eval(feed_dict={mnist.input_images:batch_xs,
    #                                               mnist.input_labels: batch_ys,
    #                                               mnist.keep_prob: 1.0})
    #     if(VALIDATION_SIZE):
    #         validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE],
    #                                                         y_: validation_labels[0:BATCH_SIZE],
    #                                                         keep_prob: 1.0})
    #         print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
    #
    #         validation_accuracies.append(validation_accuracy)
    #
    #     else:
    #          print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
    #     train_accuracies.append(train_accuracy)
    #     x_range.append(i)
    #
    #     # increase display_step
    #     if i%(display_step*10) == 0 and i:
    #         display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={mnist.input_images: batch_xs, mnist.input_labels: batch_ys, mnist.keep_prob: DROPOUT})
    if i % 5 ==0:
        result = sess.run(accuracy, feed_dict={mnist.input_images: batch_xs, mnist.input_labels: batch_ys, mnist.keep_prob: 1})
        print(result)
