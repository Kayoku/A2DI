# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Récupération des données

data = datasets.load_iris()

# Récupération des valeurs utiles

n          = len(data['data'])
nb_classes = len(data['target_names'])
nb_attr    = len(data['feature_names'])

                       
# Fonction de génération des ensembles
def generate_sets():
    x = [data['data'][0:50], data['data'][50:100], data['data'][100:150]]
    y = [data['target'][0:50], data['target'][50:100], data['target'][100:150]]
    X_app = []
    X_test = []
    Y_app = []
    Y_test = []
    
    # On prend 25 données dans les 3 classes
    for i in range(3):
        indexs = [w for w in range(50)]
        np.random.shuffle(indexs)

        for j in range(len(x[i])):
            if j < 25:
                X_app.append(x[i][indexs[j]])    
                Y_app.append(y[i][indexs[j]])
            else:
                X_test.append(x[i][indexs[j]])
                Y_test.append(y[i][indexs[j]])
    
    return X_app, Y_app, X_test, Y_test

# Création des deux ensembles d'apprentissage et 
# de test

X_app, Y_app, X_test, Y_test = generate_sets()

# Permet de renvoyer la distance entre deux exemples
def compute_distance(x, y):
    distance = 0
    for i in range(nb_attr):
        distance += (x[i] - y[i]) * (x[i] - y[i])
    return np.sqrt(distance)
    
    
# Algorithme des k plus proches voisins
def kppv(x, app, yapp, k):
    # On regarde les k plus proche voisins de x,
    # selon ses 4 attributs
    distances = []
    
    for i in range(len(app)):
        distances.append(compute_distance(x, app[i]))

    sort_index_distances = np.argsort(distances)

    # On crée le tableau des classes "les plus proches"
    # et on décide ainsi de laquelle est la plus proche
    neighbours_classes = []
    for i in range(k):
        neighbours_classes.append(yapp[sort_index_distances[i]])

    repartition = np.bincount(neighbours_classes)
    prediction = np.argmax(repartition)

    return prediction
        
kv = []
for k in range(1, 75):
    t = 0
    for i in range(len(X_test)):
        if(kppv(X_test[i], X_app, Y_app, k) == Y_test[i]):
            t += 1
    print("k: ", end='')
    print(k, end=' ')
    print(t/75)

plt.plot(kv)
