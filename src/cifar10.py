#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS671 - Deep Learning and Its Applications
Even Semester 2018

Assignment 1, Group 3
Q.2
    CIFAR-10 Classification (basic).
"""

import os
import sys
import math
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
filter_size = 7
mp_size = 2                 # mp = max pooling.
conv_stride = 1             # conv = convolution.
mp_stride = 2
fc_size = 1024              # fc = fully connected.
batch_size = 100
num_epochs = 100
num_filters = 32
num_channels = 3
num_classes = 10
num_train_batches = 5
num_images_per_batch = 10000


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



def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    """
    For batch normalization of conv_layer.
    """

    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def cnn(x, iters, is_test):
    """
    Creates the Convolutional Neural Network Model and returns the predictions.
    """

    # Weights to be used in the CNN.
    weights = {'w_conv': tf.Variable(tf.truncated_normal([filter_size, filter_size, num_channels, num_filters], stddev=0.5)),
                # Pooling reduces size by (mp_stride * mpstride).
               'w_fc': tf.Variable(tf.truncated_normal([int(image_size*image_size*num_filters/mp_stride/mp_stride), fc_size], stddev=0.5)), 
               'w_out': tf.Variable(tf.truncated_normal([fc_size, num_classes], stddev=0.5))}

    # Biases to be used in the CNN.
    biases = {'b_conv': tf.Variable(tf.truncated_normal([num_filters], stddev=0.5)),
               'b_fc': tf.Variable(tf.truncated_normal([fc_size], stddev=0.5)),
               'b_out': tf.Variable(tf.truncated_normal([num_classes], stddev=0.5))}

    conv_layer = tf.nn.conv2d(x, weights['w_conv'], strides=[1, conv_stride, conv_stride, 1], padding='SAME') + biases['b_conv']

    conv_layer, update_ema1 = batchnorm(conv_layer, is_test, iters, biases['b_conv'], convolutional=True)

    conv_layer = tf.nn.relu(conv_layer)

    conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, mp_size, mp_size, 1], strides=[1, mp_stride, mp_stride, 1], padding='SAME')

    fc = tf.reshape(conv_layer, [-1, int(image_size*image_size*num_filters/mp_stride/mp_stride)])
    fc = tf.matmul(fc, weights['w_fc']) + biases['b_fc']

    fc = tf.nn.relu(fc)

    output = tf.matmul(fc, weights['w_out']) + biases['b_out']

    return output, update_ema1


def train_cnn(iterations):
    """
    Train the CNN.
    """
    
    prediction, update_ema1 = cnn(x, iters, is_test)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    update_ema = tf.group(update_ema1)
    saver = tf.train.Saver()
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        max_learning_rate = 0.001
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        start_time = time.time()
        start = 0
        epoch = 1

        for i in range(iterations):
            
            epoch_loss = 0
            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / batch_size / decay_speed)
            
            end = start + batch_size
            batch_x = np.array(train_images[start:end])
            batch_y = np.array(train_labels[start:end])
            start = end
            
            # Training Step
            c, co, acc = sess.run([cost, correct, accuracy], feed_dict={x: batch_x, y: batch_y, is_test: False})
            epoch_loss += c

            percentage = float(end/len(train_images)) * 100
            print ("Training for this epoch: {0:.2f} %".format(round(percentage, 2)), end="")
            print (", Accuracy: {0:.2f} %".format(round(acc*100), 2), end="\r")

            if end == len(train_images):
                print('Epoch', epoch, 'completed out of '+str(num_epochs), end=' ')
                print('in {0:.2f}s, loss:'.format(round(time.time()-start_time, 2)),epoch_loss)
                print ("Testing...")
                a, c = sess.run([accuracy, cost], {x: test_images, y: test_labels, is_test: True})
                print ("Accuracy: {0:.2f}%".format(round(a*100, 2)))
                start_time = time.time()
                epoch += 1
                start = 0

            if epoch % 10 == 9:
                saver.save(sess, '../data/output/CIFAR/model/model_'+str(time.time()))

            # Backpropagation step.
            sess.run(optimizer, {x: batch_x, y: batch_y, learning_rate: lr, is_test: False})
            sess.run(update_ema, {x: batch_x, y: batch_y, is_test: False, iters: i})


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

    # Shuffle Data.
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images[:], train_labels[:] = zip(*combined)
    del combined

    # Define placeholders.
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])
    learning_rate = tf.placeholder(tf.float32)
    is_test = tf.placeholder(tf.bool)
    iters = tf.placeholder(tf.int32)

    # Calculate iterations from batch_size.
    iterations = int(num_epochs * len(train_images) / batch_size)

    # Train Data.
    print ("\nNow training...")
    train_cnn(iterations)

# End.