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
TRAINING_ITERATIONS = 500

DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10


trainPath = '../data/train.csv'
testPath = '../data/test.csv'
images, labels, test_images = dp.loadData(trainPath, testPath)


# split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
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
                        [5,5,1,32],[32],activation_func=tf.nn.relu,isDropout=False)
conv_L2 = mnist.addConvLayer(conv_L1,'conv_layer2',\
                        [5,5,32,64],[64],activation_func=tf.nn.relu,isDropout=False)

L2 = tf.reshape(conv_L2,[-1,7*7*64])

fc_L1 = mnist.addFClayer(L2, 'fc_layer1',\
                        [7*7*64, 1024],[1024],activation_func=tf.nn.relu,isDropout=True)
fc_L2 = mnist.addFClayer(fc_L1,'fc_layer2',[1024, 10],[10])
# prediction = mnist.get_prob(fc_L1, [1024, 10],[10])

#prediction是softmax的结果，所以是概率值
prediction = tf.nn.softmax(fc_L2)
predicted_labels = tf.argmax(prediction, 1)#预测出类别

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = mnist.input_labels,logits = fc_L2)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


# evaluation
# 预测对的个数
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(mnist.input_labels,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

saver = tf.train.Saver()#to save the model

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
print('loading model ...')
saver.restore(sess, '../models/lr{0}-bs{1}-cl{2}-fc{3}/model.tf'.format(LEARNING_RATE, BATCH_SIZE, 2, 2))
print('evaluating ...')
# test_prediction = predicted_labels.eval(feed_dict={mnist.input_images: test_images, mnist.keep_prob: 1.0})
# using batches is more resource efficient
validation_accuracy = accuracy.eval(feed_dict={ mnist.input_images: validation_images,
                                                            mnist.input_labels: validation_labels,
                                                            mnist.keep_prob: 1.0})
print('training_accuracy / validation_accuracy => none / %.2f for step %d'%(validation_accuracy, 1))

test_prediction = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    test_prediction[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predicted_labels.eval(feed_dict={mnist.input_images: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
                                                                                mnist.keep_prob: 1.0})
print('saving prediction ...')

sample = pd.read_csv('../data/sample_submission.csv')
sample.label = test_prediction
sample.to_csv("../outputs/mySubmission.csv", index=False)

# sess = tf.InteractiveSession()  #调用这个函数后面才可以直接使用eval函数。
# init = tf.global_variables_initializer()
# sess.run(init)
#
#
# # visualisation variables
# train_accuracies = []
# validation_accuracies = []
# x_range = []
#
# display_step=1
# for i in range(TRAINING_ITERATIONS):
#     batch_xs, batch_ys = next_batch(BATCH_SIZE)
#     # check progress on every 1st,2nd,...,10th,20th,...,100th... step
#     if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
#
#         train_accuracy = accuracy.eval(feed_dict={mnist.input_images:batch_xs,
#                                                   mnist.input_labels: batch_ys,
#                                                   mnist.keep_prob: 1.0})
#         if(VALIDATION_SIZE):
#             validation_accuracy = accuracy.eval(feed_dict={ mnist.input_images: validation_images,
#                                                             mnist.input_labels: validation_labels,
#                                                             mnist.keep_prob: 1.0})
#             print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
#
#             validation_accuracies.append(validation_accuracy)
#
#         else:
#              print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
#         train_accuracies.append(train_accuracy)
#         x_range.append(i)
#
#         # increase display_step
#         if i%(display_step*10) == 0 and i:
#             display_step *= 10
#     # train on batch
#     sess.run(train_step, feed_dict={mnist.input_images: batch_xs, mnist.input_labels: batch_ys, mnist.keep_prob: DROPOUT})
#
# print('saving model ...')
# saver.save(sess, '../models/lr{0}-bs{1}-cl{2}-fc{3}/model.tf'.format(LEARNING_RATE, BATCH_SIZE, 2, 2))
#
# sess.close()
