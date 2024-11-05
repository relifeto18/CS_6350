#!/bin/sh

echo Running Standard Perceptron, shuffling data with random seed 20
python3 StandardPerceptron.py

echo 

echo Running Voted Perceptron
python3 VotedPerceptron.py

echo 

echo Running Average Perceptron
python3 AveragePerceptron.py