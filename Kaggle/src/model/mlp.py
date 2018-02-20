import tensorflow as tf
from model.variables import *

def get_model(input_size, x):
  layer = tf.add(tf.matmul(x, weight_variable([input_size, 50])), bias_variable([1]))
  relu1 = tf.nn.relu(layer)

  layer2 = tf.add(tf.matmul(layer, weight_variable([50, 1])), bias_variable([1]))
  relu2 = tf.nn.relu(layer2)

  return relu2 
