# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:07:39 2018

@author: bouchoucha
"""
import numpy as np

nb_state = 10

def R_param(N):
    def R(s, a, sp):
        if sp == N-1 and s != N-1:
            return 100
        else:
            return 0
    
    return R
    
def T_param(N):
    def T(s, a, sp):
        if sp >= N:
            return 0
        elif a == 0 and sp - s == 1 or (sp == s == N-1):
            return 1
        elif a == 1 and sp == 0 and s != N-1:
            return 1 - (1 / ((N-1)-s))
        elif a == 1 and sp == N-1 and s != N-1:
            return (1 / ((N-1)-s))
        else:
            return 0
    return T

def possible_next_state(s):
    if s == 0:
        return [1]
    elif s == nb_state:
        return [nb_state]
    else:
        return [0, s+1, nb_state]

"""
r, t = R_param(nb_state), T_param(nb_state)

transition = [i for i in range(nb_state)]
moves = [0, 1]
possibles = [0, nb_state-1]

for tr in transition:
    for m in moves:
        for trp in possibles:
            print("({},{},{}) = {}".format(tr, m, trp, t(tr, m, trp)), end=' ')
        print("({},{},{}) = {}".format(tr, m, tr+1, t(tr, m, tr+1)))
"""       

def value_iteration(N, A, T, R, dec):
    v = [0 for _ in range(N)]
    vp = [0 for _ in range(N)]

    # On boucle 10 fois    
    for i in range(5):
        for s in range(N):
            su = []
            for a in range(A):
                su.append(0)
                for ss in range(N):
                    su[a] += T(s, a, ss) * (R(s, a, ss) + dec * v[ss])
                    print(su)
            vp[s] = np.argmax(su)
        print("----")
        v = vp.copy()

    return v # retourner fonction qui a un état, renvoie une action

policy = value_iteration(nb_state, 2, T_param(nb_state), R_param(nb_state), 0.9)