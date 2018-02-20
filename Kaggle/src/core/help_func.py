import tensorflow as tf
import numpy as np
from dataset import *

"""
Permet d'écrire le fichier de réponse
"""
def write_final_test(sess, filename, output):
  test_file = open(filename, 'w')
  test_file.write("# Id,#Class\n")
  i = 0
  while True:
    try:
      o = sess.run([output])
      test_file.write("{},{}\n".format(i,int(np.around(o))))
      i += 1
    except tf.errors.OutOfRangeError:
      break

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
Renvoie les valeurs suivantes :
 * Tableau [TN, FP, FP, TP]
 * recall      : taux de positifs détectés parmi tous les vrais positifs
 * precision   : taux de positifs détectés parmi tous les positifs détectés
 * specificity : taux de négatifs détectés parmi tous les vrais négatifs
 * accuracy    : taux de bonne classification 
"""
def test_data(sess, init_op, output, y):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  recall = 0
  precision = 0
  specificity = 0
  accuracy = 0
  cpt = 0

  sess.run(init_op) 
  while True:
    try:
      o, lbl = sess.run([output, y])#, feed_dict={handle: handle_it})
    except tf.errors.OutOfRangeError:
      break

    for i, j in zip(np.around(o), lbl):
      cpt += 1
      # Bonne détection
      if i == j:
        accuracy+=1

      # Positif
      if i == 1:
        # Vrai positif: TP
        if j == 1:
          TP+=1
        # Faux positif: FP
        else:
          FP+=1
      # Négatif
      else:
        # Faux négatif: FN
        if j == 1:
          FN+=1
        # Vrai négatif: TN
        else:
          TN+=1

  accuracy = accuracy/cpt
  recall = np.around(((TP+1) / (TP + FN + 1)) * 100, 2)
  precision = np.around(((TP+1) / (TP + FP + 1)) * 100, 2)
  specificity = np.around(((TN+1) / (FP + TN + 1)) * 100, 2)
  return [TN, FN, FP, TP], recall, precision, specificity, accuracy
