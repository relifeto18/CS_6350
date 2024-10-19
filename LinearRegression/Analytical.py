import numpy as np
import pandas as pd

# Load the training data
train_data = pd.read_csv('./data/concrete/train.csv')
X_train = train_data.iloc[:, :-1].values  # Extract features (first 7 columns)
y_train = train_data.iloc[:, -1].values  # Extract target (last column)

# Add a bias term (column of ones) to the features matrix
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term

# Calculate the optimal weight vector analytically
X_transpose = np.transpose(X_train)
optimal_weights = np.linalg.inv(X_transpose.dot(X_train)).dot(X_transpose).dot(y_train)

# Print the optimal weights
print("Optimal Weights (Analytical Solution):", optimal_weights)