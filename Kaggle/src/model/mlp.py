import tensorflow as tf
from model.variables import *

def get_model(input_size, x):
  layer = tf.add(tf.matmul(x, weight_variable([input_size, 100])), bias_variable([100]))
  relu1 = tf.nn.relu(layer)

  layer2 = tf.add(tf.matmul(relu1, weight_variable([100, 1])), bias_variable([1]))

  return layer2 
