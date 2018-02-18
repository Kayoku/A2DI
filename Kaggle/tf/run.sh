#!/bin/bash

for lr in 0.5 0.2 0.1 0.05
do
    for ba in 1000 100 10 
    do
        for ep in 100 300 500
        do
            python3 kaggle.py $ep $lr $ba
        done
    done
done
