#!/bin/sh

echo Running Batch Gradient Descent
python3 batchGD.py

echo 

echo Running Stochastic Gradient Descent
python3 SGD.py

echo 

echo Running Analytical Form
python3 Analytical.py