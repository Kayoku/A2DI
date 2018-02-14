# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:07:39 2018

@author: bouchoucha
"""
import numpy as np

#############################
# Configurations
#############################

world_size    = 20
nb_walls      = 20

gamma         = 0.9
learning_rate = 0.01
epsilon_rand  = 0.4

current_x     = 1
current_y     = 1

#############################
# Constants
#############################

possibles_actions = [0, 1, 2, 3]

#############################
# Environment creation
#############################

walls = []

for _ in range(nb_walls):
    r = np.random.randint(0, (world_size+1)*(world_size+1))
    x = r // (world_size+1)
    y = r % (world_size+1)
    walls.append((x, y)) 

for i in [0, world_size]:
    for j in range(world_size+1):
            walls.append((j, i))
            walls.append((i, j))

worlds = [['O' for _ in range(world_size+1)] for _ in range(world_size+1)]
for w in walls:
    worlds[w[0]][w[1]] = '#'

def display():
    for i in range(world_size+1):
       for j in range(world_size+1):
           print(worlds[i][j], end=' ')
       print()

#############################
# Usefull functions
#############################

def actions_str(i):
    if i == 0:
        return "bas"
    if i == 1:
        return "droite"
    if i == 2:
        return "haut"
    else:
        return "gauche"

def actions(i):
    if i == 0:
        return 1, 0
    elif i == 1:
        return 0, 1
    elif i == 2:
        return -1, 0
    else:
        return 0, -1

def reward(st):
    if st[0] == world_size-1 and st[1] == world_size-1:
        return 100
    else:
        return 0

def where_am_i(x, y):
    return np.floor(x), np.floor(y)

def move(cur_x, cur_y, x, y):
    new_x = cur_x + x
    new_y = cur_y + y
    new_x, new_y = where_am_i(new_x, new_y)
    if (new_x, new_y) not in walls:
        cur_x += x
        cur_y += y
    return cur_x, cur_y

def env(st, a):
    act = actions(a)
    sp = move(st[0], st[1], act[0], act[1])
    return sp, reward(sp)

#############################
# Création du QFunc
#############################

qfunc = {}
for st in [(i, j) for i in range(1, world_size) for j in range(1, world_size)]:
    for act in possibles_actions:
        qfunc[(st, act)] = 0

#############################
# Sarsa
#############################

def e_greedy(st):
    if np.random.random() < epsilon_rand:
        return possibles_actions[np.random.randint(0, len(possibles_actions))] 

    keys = [(st, a) for a in possibles_actions]
    values = []
    for k in keys:
        values.append(qfunc[k])
    return np.argmax(values)

def greedy(st):
    keys = [(st, a) for a in possibles_actions]
    values = []
    for k in keys:
        values.append(qfunc[k])
    return np.argmax(values)

for i in range(20000):
    current_x = 1
    current_y = 1
    s = where_am_i(current_x, current_y)
    a = e_greedy(s) 
    while not (current_x == world_size-1 and current_y == world_size-1):
        s = where_am_i(current_x, current_y)
        sp, r = env(s, a)
        ap = e_greedy(sp)
        delta = r + gamma * qfunc[(sp, ap)] - qfunc[(s, a)]
        qfunc[(s, a)] = qfunc[(s, a)] + learning_rate * delta
        current_x = sp[0]
        current_y = sp[1]
        a = ap
    print(i)

#############################
# Fitted-Q
#############################



#############################
# Play
#############################

# Fonction de jeu sur env
def put_position(x, y):
    for i in range(world_size):
        for j in range(world_size):
            if worlds[i][j] == 'I':
                worlds[i][j] = 'O'
    worlds[x][y] = 'I'

#Position de base
current_x = 1
current_y = 1
put_position(current_x, current_y)
display()

while not (current_x == world_size-1 and current_y == world_size-1):
    print()
    keys = [((current_x, current_y), a) for a in possibles_actions]
    for key in keys:
        print(key, end=' ')
        print(actions_str(key[1]), end=' ')
        print(qfunc[key])
    move_x, move_y = actions(greedy((current_x, current_y))) 
    current_x += move_x
    current_y += move_y
    put_position(current_x, current_y)
    display()
    print("--")
