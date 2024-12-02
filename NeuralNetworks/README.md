## Instructions
```
CS_6350/
├── DecisionTree/
├── EnsembleLearning/
├── LinearRegression/
├── NeuralNetworks/
│   ├── data/
│   │   └── bank-note/
│   │       └── bank-note/
│   ├── BackPropagation.py
│   ├── torch_nn.py
│   ├── SGD_NN.py
│   ├── run.sh
│   └── README.md
├── Perceptron/
├── SVM/
├── .gitignore
└── README.md
```
Please run the following command in the **Linux** system.

Please download and unzip the dataset and put them into the **data** folder under the same folder (NeuralNetworks) with python script. 

```
git clone git@github.com:relifeto18/CS_6350.git
cd CS_6350/NeuralNetworks/
mkdir data
cd data
unzip bank-note.zip -d bank-note
cd ../ (Should be under /NeuralNetworks)
chmod +x run.sh
./run.sh
```

For the back propagation problem, to change the size of hidden layer:

`python3 BackPropagation.py --hidden_layer_1_size HIDDEN_LAYER_1_SIZE --hidden_layer_2_size HIDDEN_LAYER_2_SIZE`

**To run the shell file at the current directory: `./run.sh`**
- You may need to modify the permission: `chmod +x run.sh`