import numpy as np
import pandas as pd

# Load the training and testing data
train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

# Separate features and labels
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

# Convert labels from {0, 1} to {-1, 1} for Perceptron compatibility
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Initialize weights, bias, and their averages
weights = np.zeros(X_train.shape[1])
bias = 0
avg_weights = np.zeros(X_train.shape[1])
avg_bias = 0
epochs = 10
weight_vectors = []

# Average Perceptron training
for epoch in range(epochs):
    # Training loop
    for i in range(X_train.shape[0]):
        # Calculate the linear output
        linear_output = np.dot(X_train[i], weights) + bias
        # Perceptron update rule
        if y_train[i] * linear_output <= 0:
            weight_vectors.append(weights.copy())
            weights += y_train[i] * X_train[i]
            bias += y_train[i]
        
        # Accumulate weights and bias for averaging
        avg_weights += weights
        avg_bias += bias

weight_vectors.append(weights.copy())

# Final averaged weights and bias
avg_weights /= (epochs * X_train.shape[0])
avg_bias /= (epochs * X_train.shape[0])

# Evaluate on test data using the averaged weights
predictions = np.sign(np.dot(X_test, avg_weights) + avg_bias)
test_error = np.mean(predictions != y_test)

# Display the results
print("Learned averaged weight vector:", avg_weights)
# print("Averaged bias:", avg_bias)
print("Average prediction error on the test dataset:", test_error)

# Print the list of weight vectors encountered
print("\nList of weight vectors encountered during training:")
for idx, vec in enumerate(weight_vectors):
    print(f"Update {idx+1}: {vec}")