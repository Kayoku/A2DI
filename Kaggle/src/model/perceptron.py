import tensorflow as tf
from model.variables import *

def get_model(input_size, x):
  layer = tf.add(tf.matmul(x, weight_variable([input_size, 1])), bias_variable([1]))
  return layer 
