#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS671 - Deep Learning and Its Applications
Even Semester 2018

Assignment 1, Group 3
	Input generation for Q1.
"""

import sys
from random import randint, uniform

# Integers.
file = open("../data/input/q1/input1.txt", "w")
sys.stdout = file 

for _ in range(10):
    print (str(randint(1, 9))+' '+str(randint(1, 9)))

# Float.
file = open("../data/input/q1/input2.txt", "w")
sys.stdout = file 

for _ in range(10):
    print (str(uniform(1, 9))+' '+str(uniform(1, 9)))