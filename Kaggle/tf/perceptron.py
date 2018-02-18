import tensorflow as tf
from help_func import *
from global_vars import *


weights = {
    'x' : weight_variable([NB_FEATURES, 1])
}

biases = {
    'b' : bias_variable([1])
}

def get_perceptron(x, y):
  layer = tf.squeeze(tf.matmul(x, weights['x']) + biases['b'])
  output = tf.nn.sigmoid(layer)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer))
  optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
  return output, loss, optimizer
