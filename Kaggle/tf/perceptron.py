import tensorflow as tf
from help_func import *
from global_vars import *

ALGO_NAME = "perceptron"

weights = {
    'x' : weight_variable([NB_FEATURES, 1]),
}

biases = {
    'b' : bias_variable([1]),
}

x = tf.placeholder(tf.float32, [None, NB_FEATURES], "input-data")
y = tf.placeholder(tf.float32, [None], "input-labels")

layer = tf.squeeze(tf.matmul(x, weights['x']) + biases['b'])

output    = tf.nn.sigmoid(layer)
loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


