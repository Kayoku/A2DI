# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:48:18 2018

@author: bouchoucha
"""

import h5py
import numpy as np
import tensorflow as tf
import os
import json
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from dataset import *
from help_func import *
from global_vars import *

new_epoch = int(sys.argv[1])
new_lr = float(sys.argv[2])
new_batch = int(sys.argv[3])

dic_all_values = {}
for it in range(K_FOLD):
  dt_train  = np.array(plis[0][it])
  lbl_train = np.array(plis[1][it])
  dt_test   = np.array(plis[2][it])
  lbl_test  = np.array(plis[3][it])

  tf.reset_default_graph()
  with tf.Graph().as_default(), tf.Session() as sess:

    #######################################################
    # Dataset preparation
    #######################################################

    dataset_train = tf.data.Dataset.from_tensor_slices((dt_train, lbl_train)).batch(new_batch)
    dataset_test  = tf.data.Dataset.from_tensor_slices((dt_test, lbl_test)).batch(500)

    #######################################################
    # Model
    #######################################################

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    x, y = iterator.get_next()
    x, y = tf.to_float(x), tf.to_float(y)

    layer = tf.squeeze(tf.matmul(x, weight_variable([NB_FEATURES, 1])) + bias_variable([1]))
#    relu1 = tf.nn.relu(layer)
#    layer2 = tf.squeeze(tf.matmul(relu1, weight_variable([300, 1])) + bias_variable([1]))
    output = tf.nn.sigmoid(layer)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer))
    optimizer = tf.train.AdamOptimizer(learning_rate=new_lr).minimize(loss)

    #######################################################
    # Learning
    #######################################################

    # initialisation
    training_init_op = iterator.make_initializer(dataset_train)
    test_init_op     = iterator.make_initializer(dataset_test)

    # training
    cpt = 0
    loss_mean = 0

    sess.run(tf.global_variables_initializer())
    for ep in range(new_epoch):
      sess.run(training_init_op)
      while True:
        cpt +=1
        try:
          _, l = sess.run([optimizer, loss])
          loss_mean += l
        except tf.errors.OutOfRangeError:
          break
      print("Epochs: {} loss: {} test: {}".format(ep, loss_mean/cpt, test_data(sess, output, test_init_op, y)[4]))

    #######################################################
    # Testing
    #######################################################

    all_values = test_data(sess, output, test_init_op, y)
    print("Epochs: {} loss: {} test: {}".format(new_epoch, loss_mean/cpt, all_values[4]))
    print("\nrecall: {}% precision: {}% specifity: {}%".format(all_values[1], all_values[2], all_values[3]))
    dic_all_values['perceptron'+str(it)+'-'+str(new_lr)+'-'+str(new_batch)+'-'+str(new_epoch)] = [all_values] + [loss_mean/cpt]

    #######################################################
    # Saving
    #######################################################

    save_model(sess, "save/perceptrond-"+str(new_lr)+"-"+str(new_batch)+"-"+str(new_epoch)+"/perceptron"+str(it))

with open('perceptron'+str(it)+'-'+str(new_lr)+'-'+str(new_batch)+'-'+str(new_epoch), 'w') as fp:
  json.dump(dic_all_values, fp)
