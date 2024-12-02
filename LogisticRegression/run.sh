#!/bin/sh

echo Run MAP estimation
python3 logistic_regression_map.py

echo

echo Run ML estimation
python3 logistic_regression_ml.py