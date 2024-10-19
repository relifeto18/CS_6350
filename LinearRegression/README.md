## Instructions
```
CS_6350/
├── DecisionTree/
├── EnsembleLearning/
├── LinearRegression/
│   ├── data/
│   │   └── concrete/
│   │       └── concrete/
│   ├── batchGD.py
│   ├── SGD.py
│   ├── Analytical.py
│   ├── run.sh
│   └── README.md
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (LinearRegression) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/LinearRegression/
mkdir data
cd data
unzip concrete.zip -d concrete
cd ../ (Should be under /LinearRegression)
chmod +x run.sh
./run.sh
```

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`