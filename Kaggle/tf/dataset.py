import h5py
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from global_vars import *

# Lecture du fichier de données
train  = h5py.File("../kaggle_lille1_2018_train_v1.save", 'r')
data   = np.array(train["dataset_1"])
labels = np.array(train["labels"]).astype(int)

st = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(st)
np.random.shuffle(labels)

# Lecture du fichier de test
final_test = h5py.File("../kaggle_lille1_2018_test_v1.save", 'r')
data_final = np.array(final_test["dataset_1"])

# X_input : ensemble X d'exemple
# c_input : labels de X
# k       : nombre de pli
# n_class : nombre de class
def kfold_data(X_input, c_input, k, nb_class):
    # On commence par regrouper chaque exemple
    # par leurs labels respectifs
    x_by_labels = [[] for _ in range(nb_class)]
    for i in range(len(c_input)):
        x_by_labels[c_input[i]].append(X_input[i])
    
    # On crée les différents plis selon le nombre de 
    # chacune des classes
    size_pli_by_class = [0 for _ in range(nb_class)]
    rest_pli_by_class = [0 for _ in range(nb_class)]
    
    for i in range(nb_class):
        size_pli_by_class[i] = len(x_by_labels[i]) // k
        rest_pli_by_class[i] = len(x_by_labels[i]) % k
    
    k_plis = []
    
    # Chaque boucle crée un nouveau pli
    for i in range(k):
        xx = []
        yy = []

        for j in range(nb_class):
            xx += x_by_labels[j][i*size_pli_by_class[j]:(i+1)*size_pli_by_class[j]]
            yy += [j for _ in range(size_pli_by_class[j])]
        
        s = np.random.get_state()
        np.random.shuffle(xx)
        np.random.set_state(s)
        np.random.shuffle(yy)
        
        k_plis.append([xx, yy])
    
    # On ajoute le reste de chaque classe au dernier pli
    for j in range(nb_class):
        k_plis[-1][0] += x_by_labels[j][k*size_pli_by_class[j]:]
        k_plis[-1][1] += [j for _ in range(rest_pli_by_class[j])]
        
    # Pour finir on crée toutes les possibilités de répartition des
    # plis
    plis = [[] for _ in range(4)]
    
    for i in range(len(k_plis)):
        xx = []
        yy = []
        for j in range(len(k_plis)):
            if j != i:
                xx += k_plis[j][0]
                yy += k_plis[j][1]
        plis[0].append(np.array(xx))
        plis[1].append(np.array(yy))
        plis[2].append(np.array(k_plis[i][0]))
        plis[3].append(np.array(k_plis[i][1]))
        
    return plis[0], plis[1], plis[2], plis[3]

# Réduction par threshold
#select = VarianceThreshold()
#data = select.fit_transform(data)

# Réduction par threshold
#data_final = select.fit_transform(data_final)

plis = kfold_data(data, labels, K_FOLD, 2)
