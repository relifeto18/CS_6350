## Instructions
```
CS_6350/
├── DecisionTree/
├── EnsembleLearning/
├── LinearRegression/
├── Perceptron/
├── SVM/
│   ├── data/
│   │   └── bank-note/
│   │       └── bank-note/
│   ├── run.sh
│   ├── Gaussian_kernel.py
│   ├── SVM_dual.py
│   ├── SVM_primal.py
│   └── README.md
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (SVM) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/SVM/
mkdir data
cd data
unzip bank-note.zip -d bank-note
cd ../ (Should be under /SVM)
chmod +x run.sh
./run.sh
```

You can set the random seed by using the following command line.

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`