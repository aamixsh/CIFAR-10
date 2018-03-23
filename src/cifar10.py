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
    
    return scaled_image_data, image_labels


def read_data(directory):
    """
    Read data from the downloaded directory.
    """

    # Load class names.
    print ("\nLoading Class Names...")
    class_names = [x.decode('utf-8') for x in unpickle_data(os.path.join(directory, "batches.meta"))[b'label_names']]
    print ("Done. Class Names: ")
    for i in range(len(class_names)):
        print (str(i)+". "+class_names[i])
    
    # Load training data in batches.
    train_images = np.zeros(shape=[num_train_batches*num_images_per_batch, image_size, image_size, num_channels], dtype=float)
    train_labels = np.zeros(shape=[num_train_batches*num_images_per_batch, num_classes], dtype=int)

    print ("\nLoading training images...")
    start = 0
    for i in range(num_train_batches):
        
        images_batch, label_batch = load_batch_data(os.path.join(directory, "data_batch_"+str(i+1)))
        end = start + num_images_per_batch

        # Appending batch data to the pool.
        train_images[start:end, :] = images_batch
        train_labels[start:end, :] = label_batch
        print("Done for this batch.")

        start = end

    print ("Done.")
    print ("\nNow loading test images...")
    test_images, test_labels = load_batch_data(os.path.join(directory, "test_batch"))
    print ("Done.")

    return train_images, train_labels, test_images, test_labels


def next_batch():
    """
    Returns the next batch from training data.
    """
    
    global train_images, train_labels
    # Shuffle the original list. 
    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images[:], train_labels[:] = zip(*combined)
    
    next_batch_images = []
    next_batch_labels = []
    for i in range(batch_size):
        next_batch_images.append(train_images.pop())
        next_batch_labels.append(train_labels.pop())

    return next_batch_images, next_batch_labels

def cnn(x):
    """
    Creates the Convolutional Neural Network Model.
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

    conv_layer = tf.nn.conv2d(x, weights['w_conv'], strides=[1, conv_stride, conv_stride, 1], padding='SAME')
    
    conv_layer = tf.nn.relu(conv_layer)

    # insert batch normalization here.

    conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, mp_size, mp_size, 1], strides=[1, mp_stride, mp_stride, 1], padding='SAME')
    
    fc = tf.reshape(conv_layer, [-1, int(image_size*image_size*num_filters/mp_stride/mp_stride)])
    fc = tf.matmul(fc, weights['w_fc']) + biases['b_fc']
    
    fc = tf.nn.relu(fc)

    output = tf.matmul(fc, weights['w_out']) + biases['b_out']

    return output


def train_cnn(x):
    """
    Train the CNN.
    """

    prediction = cnn(x)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    num_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(num_images_per_batch):
                epoch_x, epoch_y = next_batch()
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of '+str(num_epochs)+', loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: test_images, y: test_labels}))


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

    x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, num_channels])
    y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

    # Train Data.
    train_cnn(x)

# End.