#!/usr/bin/python3

import h5py
import numpy as np
import tensorflow as tf
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from dataset import *
from help_func import *
from global_vars import *
from perceptron import *
#from kaggle import *

if len(sys.argv) == 1:
  print("usage: ./generate_test.py save-name")
  exit(-1)

filename = sys.argv[1]

with tf.Session() as sess:
  dataset_final = tf.data.Dataset.from_tensor_slices((data_final, tf.ones_like(data_final))).batch(1)

  iterator = tf.data.Iterator.from_structure(dataset_final.output_types, dataset_final.output_shapes)
  x, y = iterator.get_next()
  x, y = tf.to_float(x), tf.to_float(y)

  layer = tf.squeeze(tf.matmul(x, weights['x']) + biases['b'])
  output = tf.nn.sigmoid(layer)

#  output, _, _ = get_perceptron(x, y)

  final_init_op = iterator.make_initializer(dataset_final)

  restore_model(sess, "save/"+filename)
  write_final_test(sess, "test.csv", output, final_init_op) 
