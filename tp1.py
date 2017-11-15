# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from sklearn import datasets
import numpy as np

# Récupération des données

data = datasets.load_iris()

# Récupération des valeurs utiles

n          = len(data['data'])
nb_classes = len(data['target_names'])
nb_attr    = len(data['feature_names'])

# Création des deux ensembles d'apprentissage et 
# de test

D_app  = np.concatenate((data['data'][0:25],
                        data['data'][50:75],
                        data['data'][100:125]))
D_test = np.concatenate((data['data'][25:50],
                        data['data'][75:100],
                        data['data'][125:150]))

Y_app  = np.concatenate((data['target'][0:25],
                        data['target'][50:75],
                        data['target'][100:125]))
Y_test = np.concatenate((data['target'][25:50],
                        data['target'][75:100],
                        data['target'][125:150]))

# Permet de renvoyer la distance entre deux exemples
def compute_distance(x, y):
    distance = 0
    for i in range(nb_attr):
        distance += (x[i] - y[i]) * (x[i] - y[i])
    return np.sqrt(distance)
    
    
# Algorithme des k plus proches voisins
def kppv(x, app, k):
    # On regarde les k plus proche voisins de x,
    # selon ses 4 attributs
    distances = []
    for i in range(len(app)):
        distances.append(compute_distance(x, app[i]))

    sort_index_distances = np.argsort(distances)

    # On regarde les classes de ces 4 attributs et
    # on prend la plus dominante
    neighbours_classes = []
    for i in range(k):
        neighbours_classes.append(Y_app[sort_index_distances[i]])

    repartition = np.bincount(neighbours_classes)
    prediction = np.argmax(repartition)
    return prediction

    # Si il y a une égalité, on fait la diff
    # entre les sommes
    #if np.count_nonzero(repartition == prediction) > 1: