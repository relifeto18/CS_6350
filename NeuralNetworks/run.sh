#!/bin/sh

echo Run back propagation algorithm for the entire dataset
echo Default hidden layer side is 2
echo Only output the gradients for the last row of data
python3 BackPropagation.py

echo

echo Run stochastic gradient descent algorithm
python3 SGD_NN.py

echo    

echo Run PyTorch version
python3 torch_nn.py