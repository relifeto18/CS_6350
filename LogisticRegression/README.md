## Instructions
```
CS_6350/
├── DecisionTree/
├── EnsembleLearning/
├── LinearRegression/
├── LogisticRegression/
│   ├── data/
│   │   └── bank-note/
│   │       └── bank-note/
│   ├── logistic_regression_map.py
│   ├── logistic_regression_ml.py
│   ├── run.sh
│   └── README.md
├── NeuralNetworks/
├── Perceptron/
├── SVM/
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (LogisticRegression) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/LogisticRegression/
mkdir data
cd data
unzip bank-note.zip -d bank-note
cd ../ (Should be under /LogisticRegression)
chmod +x run.sh
./run.sh
```

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`