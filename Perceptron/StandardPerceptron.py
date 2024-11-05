import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Standard Perceptron for binary classification.")
parser.add_argument("--seed", type=int, default=20, help="Random seed for reproducibility (default: 20).")

args = parser.parse_args()

# Load the training and testing data
train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

# Separate features and labels
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

# Convert labels from {0, 1} to {-1, 1} for Perceptron compatibility
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Initialize weights and parameters
weights = np.zeros(X_train.shape[1])
bias = 0
epochs = 10
np.random.seed(args.seed)

# Perceptron training
for epoch in range(epochs):
    # Shuffle the training data at the start of each epoch
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    
    for i in range(X_train.shape[0]):
        # Calculate the linear output
        linear_output = np.dot(X_train[i], weights) + bias
        # Perceptron update rule
        if y_train[i] * linear_output <= 0:
            weights += y_train[i] * X_train[i]
            bias += y_train[i]

# Evaluate on test data
predictions = np.sign(np.dot(X_test, weights) + bias)
test_error = np.mean(predictions != y_test)

# Display the results
print("Learned weight vector:", weights)
# print("Bias:", bias)
print("Average prediction error on the test dataset:", test_error)