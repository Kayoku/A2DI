import tensorflow as tf
import numpy as np
from dataset import *

"""
Permet de créer les poids
"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

"""
Permet de créer les biais
"""
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
Permet d'écrire le fichier de réponse
"""
def write_final_test(filename, output, x, y):
  test_file = open(filename, 'w')
  test_file.write("# Id,#Class\n")
  for i in range(len(data_final)):
      o = output.eval({x: [data_final[i]]})
      test_file.write("{},{}\n".format(i,int(np.around(o))))
  test_file.close()

"""
Permet de sauver le modèle actuel
"""
def save_model(sess, name):
  saver = tf.train.Saver()
  done = saver.save(sess, name)
  print("Model save in: {}".format(done))

"""
Permet de recharger un modèle
"""
def restore_model(sess, name):
  saver = tf.train.Saver()
  saver.restore(sess, name)
  print("Model {} restored.".format(name))

"""
Evalue la précision sur des données du modèle
"""
def test_data(output, x, y, dt, lbl):
  m = 0
  for i in range(len(dt)):
    o = output.eval({x: [dt[i]], y: [lbl[i]]})
    if np.around(o) == lbl[i]:
      m+=1
  m = m/len(dt)
  return m
