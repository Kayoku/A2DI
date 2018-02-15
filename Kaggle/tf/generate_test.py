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

if len(sys.argv) == 1:
  print("usage: ./generate_test.py save-name")
  exit(-1)

filename = sys.argv[1]

with tf.Session() as sess:
  restore_model(sess, "save/"+filename)
  write_final_test("test.csv", output, x, y) 
