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
Renvoie les valeurs suivantes :
 * Tableau [TN, FP, FP, TP]
 * recall      : taux de positifs détectés parmi tous les vrais positifs
 * precision   : taux de positifs détectés parmi tous les positifs détectés
 * specificity : taux de négatifs détectés parmi tous les vrais négatifs
 * accuracy    : taux de bonne classification 
"""
def test_data(output, x, y, dt, lbl):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  recall = 0
  precision = 0
  specificity = 0
  accuracy = 0

  for i in range(len(dt)):
    o = output.eval({x: [dt[i]], y: [lbl[i]]})
    v = np.around(o)

    # Bonne détection
    if v == lbl[i]:
      accuracy+=1

    # Positif
    if v == 1:
      # Vrai positif: TP
      if lbl[i] == 1:
        TP+=1
      # Faux positif: FP
      else:
        FP+=1
    # Négatif
    else:
      # Faux négatif: FN
      if lbl[i] == 1:
        FN+=1
      # Vrai négatif: TN
      else:
        TN+=1

  accuracy = accuracy/len(dt)
  recall = np.around((TP / (TP + FN)) * 100, 2)
  precision = np.around((TP / (TP + FP)) * 100, 2)
  specificity = np.around((TN / (FP + TN)) * 100, 2)
  return [TN, FN, FP, TP], recall, precision, specificity, accuracy

"""
Renvoie le taux de bonne classification
"""
def test_data_accuracy(output, x, y, dt, lbl):
  accuracy = 0
  for i in range(len(dt)):
    o = output.eval({x: [dt[i]], y: [lbl[i]]})
    if np.around(o) == lbl[i]:
      accuracy+=1

  accuracy = accuracy / len(dt)
  return accuracy
