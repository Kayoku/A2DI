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

"""
Variable utiles
"""
nb_features   = 4004
learning_rate = 0.3
nb_batch      = 100
epochs        = 2000

"""
Fonctions utiles
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

weights = {
    'x' : weight_variable([nb_features, 1]),
}

biases = {
    'b' : bias_variable([1]),
}

"""
Réseau
"""

x = tf.placeholder(tf.float32, [None, nb_features], "input-data")
y = tf.placeholder(tf.float32, [None], "input-labels")

layer = tf.squeeze(tf.matmul(x, weights['x']) + biases['b'])

output = tf.nn.sigmoid(layer)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

"""
Train
"""
with tf.Session() as sess:
    sess.run(init)
    
    for ep in range(epochs):
        cost_mean = 0
        X_batches = np.array_split(data, nb_batch)
        Y_batches = np.array_split(labels, nb_batch)
        
        for b in range(nb_batch):
            batch_x, batch_y = X_batches[b], Y_batches[b]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            
            cost_mean += c/len(X_batches[b])
        
        if (ep%10==0):
            print("Epochs: {} cost: {}".format(ep+1, cost_mean))
    
    m = 0
    for i in range(len(data)):
        o = output.eval({x: [data[i]], y: [labels[i]]})
        if o > 0.5 and labels[i] == 1. or o < 0.5 and labels[i] == 0.:
            m +=1
    m = m/len(data)
    print("Accuracy: {}".format(m))

    # Test
    test_file = open("test.csv", 'w')
    test_file.write("# Id,#Class\n")
    for i in range(len(data_final)):
        o = output.eval({x: [data_final[i]]})
        test_file.write("{},{}\n".format(i,int(np.around(o))))
    test_file.close()
