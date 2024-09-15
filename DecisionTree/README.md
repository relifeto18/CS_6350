## Instructions
```
CS_6350/
├── DecisionTree/
│   ├── data/
│   │   ├── car/
│   │   └── bank/
│   ├── ID3_bank.py
│   ├── ID3_car.py
│   ├── run.sh
│   └── README.md
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (DecisionTree) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/DecisionTree/
mkdir data
cd data
unzip car.zip -d car
unzip bank.zip -d bank
cd ../ (Should be under /DecisionTree)
chmod +x run.sh
./run.sh
```

You will be asked to choose a maximum depth for the tree for the first task (car).
- The depth is between **1 - 6**.

You will be asked to choose a way to treat unknown values and a maximum depth for the tree for the first task (bank).
 - The treatment could be either **0** (treat as a particular attribute value) or **1** (treat as missing attribute value).
- The depth is between **1 - 16**.

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`

(Optional) Usage of the python script: `python ID3_car.py` `python ID3_bank.py`