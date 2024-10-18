#!/bin/sh

echo Running AdaBoost
python3 AdaBoost.py

echo 

echo Running Bagging
python3 Bagging.py

echo Running Bagging Comparison
python3 Bagging_comparison.py

echo 

echo Running Random Forest
python3 RandomForest.py

echo Running Random Forest Comparison
python3 RandomForest_comparison.py

# echo 

# echo Running Bonus
# python3 Credit.py