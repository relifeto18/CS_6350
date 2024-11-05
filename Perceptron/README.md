## Instructions
```
CS_6350/
├── DecisionTree/
├── EnsembleLearning/
├── LinearRegression/
├── Perceptron/
│   ├── data/
│   │   └── bank-note/
│   │       └── bank-note/
│   ├── AveragePerceptron.py
│   ├── StandardPerceptron.py
│   ├── VotedPerceptron.py
│   ├── run.sh
│   └── README.md
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (Perceptron) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/Perceptron/
mkdir data
cd data
unzip bank-note.zip -d bank-note
cd ../ (Should be under /Perceptron)
chmod +x run.sh
./run.sh
```

You can set the random seed by using the following command line.

`python StandardPerceptron.py --seed 20`

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`