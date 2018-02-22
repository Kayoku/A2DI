import tensorflow as tf
from model.variables import *

def get_model(input_size, x):
  layer = tf.add(tf.matmul(x, weight_variable([input_size, 100])), bias_variable([100]))
  relu1 = tf.nn.relu(layer)

  layer2 = tf.add(tf.matmul(relu1, weight_variable([100, 10])), bias_variable([10]))
  relu2 = tf.nn.relu(layer2)

  layer3 = tf.add(tf.matmul(relu2, weight_variable([10, 1]), bias_variable([1])))

  return layer3
