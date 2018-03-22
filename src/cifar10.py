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
import download
import scipy.misc
import numpy as np
import tensorflow as tf
# from dataset import one_hot_encoded

input_data_path = "../data/input/CIFAR"
output_data_path = "../data/output/CIFAR"
image_size = 32
num_channels = 3
num_classes = 10
num_train_batches = 5
num_images_per_batch = 10000


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


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
	Store data from array as image for visualization.
	"""

	# Reshape flattened data into RGB format.
	x = arr.reshape([-1, num_channels, image_size, image_size])
	x = np.transpose(x, (0, 2, 3, 1))
	
	path = os.path.join(output_data_path, "images")
	create_path(path)
	for i in range(x):
		# Save
		scipy.misc.imsave(os.path.join(path, "output"+str(i+1)+".jpg"), x[i])


def load_batch_data(filename):
	"""
	Load data from single batch.
	"""

	raw_data = unpickle_data(filename)
	image_labels = np.array(raw_data[b'labels'])
	image_data = raw_data[b'data']
	
	# Rescale data between 0-1
	scaled_image_data = np.array(image_data, dtype=float) / 255
	
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
	images = np.zeros(shape=[num_train_batches*num_images_per_batch, image_size*image_size*num_channels], dtype=float)
	labels = np.zeros(shape=[num_train_batches*num_images_per_batch], dtype=int)

	print ("Loading images...")
	start = 0
	for i in range(num_train_batches):
		
		images_batch, label_batch = load_batch_data(os.path.join(directory, "data_batch_"+str(i+1)))
		end = start + num_images_per_batch

		# Appending batch data to the pool.
		images[start:end, :] = images_batch
		labels[start:end] = label_batch
		print("Done for this batch.")

		start = end

	return images, labels


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
	download.maybe_download_and_extract(url=url, download_dir=input_data_path)

	# Unpickle data.
	images, labels = read_data(os.path.join(input_data_path,"cifar-10-batches-py"))

	
# End.

