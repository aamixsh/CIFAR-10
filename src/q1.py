#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS671 - Deep Learning and Its Applications
Even Semester 2018

Assignment 1, Group 3
Q.1
	Using tensorflow for a simple computation.
"""

import os
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Creates subdirectories if not present in a path.
def createPath(output):
	if not os.path.exists(os.path.dirname(output)):
		try:
			os.makedirs(os.path.dirname(output))
		except OSError as exc:
			if exc.errorno!=errorno.EEXIST:
				raise

# Main function.
if __name__ == '__main__':
	
	# Input.
	a, b = [], []

	choice = input("Want to use own input file or default? (o/d): ")

	# Read input file.
	if choice in ['d', 'D']:
		file = open("../data/input/q1/input1.txt")
	else:
		path = input("Input file path: ")
		file = open(path)
		filename = path.split('/')[-1].split('.')[0]

	for line in file:
		[x, y] = [float(n) for n in line.split()]
		a.append(x)
		b.append(y)

	# Use tensors.
	a = tf.convert_to_tensor(a)
	b = tf.convert_to_tensor(b)
	c = (a+b) * (b+1)

	# Run session.
	sess = tf.Session()
	c = sess.run(c)

	# Write output to file.
	path = "../data/output/q1/"
	if choice in ['d', 'D']:
		path += "output_input1.txt"
	else:
		path += "output_"+filename+".txt"

	createPath(path)

	file = open(path, "w")
	sys.stdout = file
	for i in range(len(c)):
		print (c[i])

# End.

