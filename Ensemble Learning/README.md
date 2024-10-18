## Instructions
```
CS_6350/
├── DecisionTree/
├── Ensemble Learning/
│   ├── data/
│   │   ├── bank/
│   │   └── credit_card/ (Optional)
│   ├── AdaBoost.py
│   ├── Bagging.py
│   ├── Bagging_comparison.py
│   ├── Credit.py (Optional)
│   ├── RandomForest.py
│   ├── RandomForest_comparison.py
│   ├── run.sh
│   └── README.md
├── Linear Regression/
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (Ensemble Learning) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/Ensemble Learning/
mkdir data
cd data
unzip bank.zip -d bank
cd ../ (Should be under /Ensemble Learning)
chmod +x run.sh
./run.sh
```

Please wait for some time to allow the code executed. 

To use parallel threads for speed up, `pip install joblib` and `pip install tqdm`. The parameter `n_jobs` is the number of CPU cores. 

For question 2(b), `Bagging.py`

For question 2(c), `Bagging_comparison.py`

For question 2(d), `RandomForest.py`

For question 2(e), `RandomForest_comparison.py`

The file `Credit.py` is for Bonus question, which is not completed.

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`