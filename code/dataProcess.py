# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import tensorflow as tf

IMAGE_TO_DISPLAY = 10

def loadData(trainPath,testPath):
    trainData = pd.read_csv(trainPath)
    testData = pd.read_csv(testPath)

    # print('data({0[0]},{0[1]})'.format(data.shape))
    # print (data.head())

    train_images = trainData.iloc[:,1:].values.astype(np.float)
    test_images = testData.values.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    train_images = np.multiply(train_images, 1.0 / 255.0)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    #if you want to display an image for fun, call the function below
    display(test_images[IMAGE_TO_DISPLAY])

    #------------------------------------------------------#
    labels = trainData.iloc[:,0].values
    labels = labels_to_one_hot(labels)

    return train_images, labels, test_images


def display(img):
    image_size = len(img)
    # in this case all images are square
    width = height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    one_imge = img.reshape(width,height)
    plt.axis('off')
    plt.imshow(one_imge, cmap=cm.binary)



def labels_to_one_hot(labels):
    num_classes = np.unique(labels).shape[0]
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    #flat函数讲矩阵看成一维数组，下标就是按行索引
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot.astype(np.uint8)




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
    
    args, unparsed = parser.parse_known_args()
    #trainPath = '../data/train.csv'
    #testPath = '../data/test.csv'
    train_images, train_labels, test_images = loadData(args.trainPath, args.testPath)
    
    print(train_images.shape)
    print(test_images.shape)
    print(train_labels.shape)
    
    features_placeholder = tf.placeholder(train_images.dtype, train_images.shape)
    labels_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.batch(4).repeat()
    iterator = dataset.make_initializable_iterator()
    #iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
   
    sess = tf.Session()
    
    sess.run(iterator.initializer, feed_dict={features_placeholder: train_images,
                                          labels_placeholder: train_labels})
    while(1):
        a,b = sess.run(next_element)
    print(a.shape, b.shape)
    plt.show()
