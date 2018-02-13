# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:00:08 2018

@author: bouchoucha
"""

f = open("test.csv", 'w')
a = 0
for i in range(4000):
    if np.random.random() > 0.5:
        a = 1
    else:
        a = 0
    f.write("{},{}\n".format(i, a))
f.close()