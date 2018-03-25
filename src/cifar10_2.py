#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS671 - Deep Learning and Its Applications
Even Semester 2018

Assignment 1, Group 3
Q.2
    CIFAR-10 Classification.
"""

import os
import sys
import time
import pickle
import urllib
import random
import tarfile
import scipy.misc
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

input_data_path = "../data/input/CIFAR"
output_data_path = "../data/output/CIFAR"
image_size = 32
filter_size = 5
filter_size2 = 3
mp_size = 2                 # mp = max pooling.
conv_stride = 1             # conv = convolution.
mp_stride = 2
fc_size = 1024              # fc = fully connected.
fc_size2 = 4096
batch_size = 100
num_epochs = 100
num_filters = 32
num_channels = 3
num_classes = 10
num_train_batches = 5
num_images_per_batch = 10000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Code starts here.

def create_path(output):
    """
    Create subdirectories if not present in a path.
    """

    if not os.path.exists(output):
        try:
            os.makedirs(output)
        except OSError as exc:
            if exc.errorno!=errorno.EEXIST:
                raise


def maybe_download_and_extract(url, dest_directory):
    """
    Download and extract data from url. 
    Function taken from "https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py"
    and modified.
    """
  
    create_path(dest_directory)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            # Show download progress.
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, 
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        print ("Downloading data...")
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(filepath)
        
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
        print ("Extracting data...")
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        print ("Done.")
    else:
        print ("Data already downloaded and extracted.")


def unpickle_data(filename):
    """
    Unpickle Data.
    """

    print ("Loading data from file:", filename)
    file = open(filename, "rb")
    data = pickle.load(file, encoding="bytes")

    return data


def imagify(arr):
    """
    Store data from array as images for visualization.
    """
    
    path = os.path.join(output_data_path, "images")
    create_path(path)
    for i in range(x):
        scipy.misc.imsave(os.path.join(path, "output"+str(i+1)+".jpg"), x[i])


def load_batch_data(filename):
    """
    Load data from single batch.
    """

    raw_data = unpickle_data(filename)
    image_labels = np.array(raw_data[b'labels'])
    
    # One-Hot Encoding.
    temp = np.zeros(shape=[num_images_per_batch, num_classes])
    for i in range(num_images_per_batch):
        temp[i][image_labels[i]] = 1
    image_labels = temp
    
    image_data = raw_data[b'data']
    
    # Rescale data between 0-1
    scaled_image_data = np.array(image_data, dtype=float) / 255
    x = scaled_image_data.reshape([-1, num_channels, image_size, image_size])
    scaled_image_data = np.transpose(x, (0, 2, 3, 1))
    
    # Making lists from np arrays.
    scaled_image_data = [x for x in scaled_image_data]
    image_labels = [x for x in image_labels]

    return scaled_image_data, image_labels


def read_data(directory):
    """
    Read data from the downloaded directory.
    """

    # Load class names.
    print ("\nLoading Class Names...")
    class_names = [x.decode('utf-8') for x in unpickle_data(os.path.join(directory, "batches.meta"))[b'label_names']]
    print ("Done. Class Names: \n")
    for i in range(len(class_names)):
        print (str(i)+". "+class_names[i])
    
    # Load training data in batches.
    train_images = []
    train_labels = []

    print ("\nLoading training images...")
    start = 0
    for i in range(num_train_batches):
        
        images_batch, label_batch = load_batch_data(os.path.join(directory, "data_batch_"+str(i+1)))

        # Appending batch data to the pool.
        train_images.extend(images_batch)
        train_labels.extend(label_batch)
        print("Done for this batch.")

    print ("Done.")
    print ("\nNow loading test images...")
    test_images, test_labels = load_batch_data(os.path.join(directory, "test_batch"))
    print ("Done.")

    return train_images, train_labels, test_images, test_labels


# def initialize_batch_norm(scope, depth):
#     with tf.variable_scope(scope) as bnscope:
#         gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
#         beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
#         moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
#         moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
#         bnscope.reuse_variables()


# def BatchNorm_layer(x, scope, train, epsilon=0.001, decay=.99):
#     # Perform a batch normalization after a conv layer or a fc layer
#     # gamma: a scale factor
#     # beta: an offset
#     # epsilon: the variance epsilon - a small float number to avoid dividing by 0
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#         with tf.variable_scope('BatchNorm', reuse=True) as bnscope:
#             gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
#             moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
#             shape = x.get_shape().as_list()
#             control_inputs = []
#             if train:
#                 avg, var = tf.nn.moments(x, range(len(shape)-1))
#                 update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
#                 update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
#                 control_inputs = [update_moving_avg, update_moving_var]
#             else:
#                 avg = moving_avg
#                 var = moving_var
#             with tf.control_dependencies(control_inputs):
#                 output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
#     return output


def cnn(x, is_training):
    """
    Creates the Convolutional Neural Network Model.
    """

    # Weights to be used in the CNN.
    weights = { 'w_conv': tf.Variable(tf.truncated_normal([filter_size, filter_size, num_channels, num_filters], stddev=0.5)),
                'w_conv2': tf.Variable(tf.truncated_normal([filter_size2, filter_size2, num_filters, num_filters], stddev=0.5)),
                'w_conv3': tf.Variable(tf.truncated_normal([filter_size2, filter_size2, num_filters, num_filters], stddev=0.5)),
                # Pooling reduces size by (mp_stride * mpstride).
                'w_fc': tf.Variable(tf.truncated_normal([int(image_size*image_size*num_filters/mp_stride**2/mp_stride**2), fc_size], stddev=0.5)), 
                'w_fc2': tf.Variable(tf.truncated_normal([fc_size, fc_size2], stddev=0.5)), 
                'w_out': tf.Variable(tf.truncated_normal([fc_size2, num_classes], stddev=0.5))}
    
    # Biases to be used in the CNN.
    biases = {  'b_conv': tf.Variable(tf.truncated_normal([num_filters], stddev=0.5)),
                'b_conv2': tf.Variable(tf.truncated_normal([num_filters], stddev=0.5)),
                'b_conv3': tf.Variable(tf.truncated_normal([num_filters], stddev=0.5)),
                'b_fc': tf.Variable(tf.truncated_normal([fc_size], stddev=0.5)),
                'b_fc2': tf.Variable(tf.truncated_normal([fc_size2], stddev=0.5)),
                'b_out': tf.Variable(tf.truncated_normal([num_classes], stddev=0.5))}

    conv_layer = tf.nn.conv2d(x, weights['w_conv'], strides=[1, conv_stride, conv_stride, 1], padding='SAME') + biases['b_conv']
    
    conv_layer = tf.nn.relu(conv_layer)

    conv_layer = tf.nn.lrn(conv_layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='conv_layer')

    conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, mp_size, mp_size, 1], strides=[1, mp_stride, mp_stride, 1], padding='SAME')
    
    conv_layer2 = tf.nn.conv2d(conv_layer, weights['w_conv2'], strides=[1, conv_stride, conv_stride, 1], padding='SAME') + biases['b_conv2']

    conv_layer2 = tf.nn.relu(conv_layer2)

    conv_layer2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='conv_layer2')

    conv_layer3 = tf.nn.conv2d(conv_layer2, weights['w_conv3'], strides=[1, conv_stride, conv_stride, 1], padding='SAME') + biases['b_conv3']

    conv_layer3 = tf.nn.relu(conv_layer3)

    conv_layer3 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='conv_layer3')

    conv_layer3 = tf.nn.max_pool(conv_layer3, ksize=[1, mp_size, mp_size, 1], strides=[1, mp_stride, mp_stride, 1], padding='SAME')

    fc = tf.reshape(conv_layer3, [-1, int(image_size*image_size*num_filters/mp_stride**2/mp_stride**2)])

    fc = tf.matmul(fc, weights['w_fc']) + biases['b_fc']
    
    fc = tf.nn.relu(fc)

    fc2 = tf.matmul(fc, weights['w_fc2']) + biases['b_fc2']
    
    fc2 = tf.nn.relu(fc2)

    output = tf.matmul(fc2, weights['w_out']) + biases['b_out']

    return output


def train_cnn(x, is_training, learning_rate=0.0001):
    """
    Train the CNN.
    """

    prediction = cnn(x, is_training)
        
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            
            epoch_loss = 0
            i = 0
            start_time = time.time()

            while i < len(train_images):
                start = i
                end = i + batch_size

                epoch_x = np.array(train_images[start:end])
                epoch_y = np.array(train_labels[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, is_training: True})
                epoch_loss += c

                percentage = float(end/num_train_batches/num_images_per_batch)*100
                print ("Training for this epoch: {0:.2f}%".format(round(percentage, 2)), end="\r")

                i = end

            # saver = tf.train.Saver()
            # saver.save(sess, '../data/output/CIFAR/model/model_'+str(time.time()))

            print('Epoch', epoch+1, 'completed out of '+str(num_epochs), end=' ')
            print('in {0:.2f}s, loss:'.format(round(time.time()-start_time, 2)),epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print ("Testing...")
            i = 0
            n = 0
            acc = 0.0
            while i < len(test_images):
                start = i
                end = i + batch_size

                epoch_x = np.array(test_images[start:end])
                epoch_y = np.array(test_labels[start:end])
                acc += accuracy.eval({x: epoch_x, y: epoch_y, is_training: False})

                print ("Accuracy: ", acc, end='\r') 
                i = end
                n += 1

            print ("Accuracy: ", acc)


if __name__ == '__main__':
    """
    Main function.
    """

    print("This program will train the CNN for CIFAR-10 dataset.")

    choice = input("Want to use own input download URL or default (will not download again if already done)? (o/d): ")

    # Read input.
    if choice in ['d', 'D']:
        url = "https://students.iitmandi.ac.in/~s16007/cifar-10-python.tar.gz"
    else:
        url = input("Input download URL: ")
    
    create_path(input_data_path)
    create_path(output_data_path)

    # Code used from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/download.py
    # Download and extract data from the URL.
    maybe_download_and_extract(url=url, dest_directory=input_data_path)

    # Unpickle Data.
    train_images, train_labels, test_images, test_labels = read_data(os.path.join(input_data_path,"cifar-10-batches-py"))
    # test_images = np.array(test_images)
    # test_labels = np.array(test_labels)

    # Shuffle Data.
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images[:], train_labels[:] = zip(*combined)
    del combined

    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    is_training = tf.placeholder(tf.bool)

    # Train Data.
    print ("\nNow training...")
    train_cnn(x, is_training)

# End.