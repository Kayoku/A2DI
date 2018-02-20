import tensorflow as tf
import numpy as np
import os
import sys
import json

from dataset.dataset import *
from core.help_func import *

import model.perceptron as perceptron 
import model.mlp as mlp

def usage():
  print("usage: python3 kaggle learning pre model lr bash epochs\n       python3 kaggle test pre model name")
  exit(-1)

if len(sys.argv) < 4:
  usage() 

#######################
# Pré-traitement
#######################

pre = sys.argv[2]
model = sys.argv[3]

if pre == 'variances':
  training_data, validation_data, final_data = variance_filter(0.0001)
  print("Variance reduction.")
else:
  print("No reduction.")

print("Nb de dimensions: {}".format(np.shape(training_data)[1]))

#################################################
# Learning
#################################################

if sys.argv[1] == 'learning' and len(sys.argv) == 7:
  lr = float(sys.argv[4])
  batch = int(sys.argv[5])
  epochs = int(sys.argv[6])

  ########################
  # Dataset preparation
  ########################

  dataset_train = tf.data.Dataset.from_tensor_slices((training_data, training_labels)).batch(batch)
  dataset_validation = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels)).batch(100)

  iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
  x, y = iterator.get_next()
  x, y = tf.to_float(x), tf.to_float(y)

  training_init_op = iterator.make_initializer(dataset_train)
  validation_init_op = iterator.make_initializer(dataset_validation)

  ########################
  # Model
  ########################

  if model == 'perceptron':
    model_output = perceptron.get_model(np.shape(training_data)[1], x)
  elif model == 'mlp':
    model_output = mlp.get_model(np.shape(training_data)[1], x)

  output = tf.nn.sigmoid(model_output)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.squeeze(model_output)))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

  ########################
  # Learning step 
  ########################

  all_values = {}
  with tf.Session() as sess:
    cpt = 0
    loss_mean = 0

    sess.run(tf.global_variables_initializer())

    # Epoch step
    for ep in range(epochs):
      sess.run(training_init_op)
      while True:
        cpt += 1
        try:
          _, l = sess.run([optimizer, loss])
          loss_mean += l
        except tf.errors.OutOfRangeError:
          break
      all_values = test_data(sess, validation_init_op, output, y)
      print("Epochs: {} loss: {} test: {}".format(ep, loss_mean/cpt, all_values[4]))

    ########################
    # Saving
    ########################
  
    save_model(sess, "save/"+model+"-"+pre+"-"+str(lr)+"-"+str(batch)+"-"+str(epochs)+"/"+model)
    with open("save/"+model+'-'+pre+'-'+str(lr)+'-'+str(batch)+'-'+str(epochs)+'/'+model+'.json', 'w') as fp:
      json.dump(all_values, fp)

#################################################
# Test
#################################################

elif sys.argv[1] == 'test' and len(sys.argv) == 5:
  filename = sys.argv[4]

  ########################
  # Dataset
  ########################

  dataset_final = tf.data.Dataset.from_tensor_slices(final_data).batch(1)
  iterator = dataset_final.make_one_shot_iterator()
  x = tf.to_float(iterator.get_next())

  ########################
  # Model
  ########################

  if model == 'perceptron':
    model_output = perceptron.get_model(np.shape(training_data)[1], x)
  elif model == 'mlp':
    model_output = mlp.get_model(np.shape(training_data)[1], x)

  output = tf.nn.sigmoid(model_output)

  ########################
  # Restore model and test
  ########################

  with tf.Session() as sess:
    restore_model(sess, "save/"+filename)
    write_final_test(sess, "test.csv", output)

else:
  usage()
