# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:48:18 2018

@author: bouchoucha
"""

import h5py
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']="3"

from dataset import *
from help_func import *
from global_vars import *
from perceptron import *

for it in range(K_FOLD):
  dt_train  = plis[0][it]
  lbl_train = plis[1][it]
  dt_test   = plis[2][it]
  lbl_test  = plis[3][it]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #######################################################
    # Learning
    #######################################################

    for ep in range(EPOCHS):
      loss_mean = 0
      X_batches = np.array_split(dt_train, NB_BATCH)
      Y_batches = np.array_split(lbl_train, NB_BATCH)
      
      for b in range(NB_BATCH):
        batch_x, batch_y = X_batches[b], Y_batches[b]
        _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

        loss_mean += c/len(X_batches[b])

      if (ep%100==0):
        print("Epochs: {} loss: {} test: {}".format(ep+1, loss_mean, test_data(output, x, y, dt_test, lbl_test)))

    #######################################################
    # Testing
    #######################################################

    print("Accuracy on test: {}".format(test_data(output, x, y, dt_test, lbl_test)))
    print("Accuracy on train: {}".format(test_data(output, x, y, dt_train, lbl_train)))

    #######################################################
    # Saving
    #######################################################

    save_model(sess, "save/"+ALGO_NAME+str(it))

