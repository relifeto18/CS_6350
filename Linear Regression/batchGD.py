#!/usr/bin/env python

import numpy as np
import pandas as pd

# Load the training data
train_data = pd.read_csv('./data/concrete/train.csv')
X_train = train_data.iloc[:, :-1].values  # Extract features (first 7 columns)
y_train = train_data.iloc[:, -1].values  # Extract target (last column)

test_data = pd.read_csv('./data/concrete/test.csv')
X_test = test_data.iloc[:, :-1].values  # Extract features (first 7 columns)
y_test = test_data.iloc[:, -1].values

# Add a bias term (column of ones) to the features matrix
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Function to compute the cost (Mean Squared Error)
def compute_cost(X, y, weights):
    n_samples = len(y)
    predictions = np.dot(X, weights)
    errors = predictions - y
    cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
    return cost

# Define the batch gradient descent function
def batch_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, max_iters=10000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to zeros
    cost_history = []
    
    for iteration in range(max_iters):
        # Compute the prediction
        predictions = np.dot(X, weights)
        
        # Compute the error
        errors = predictions - y
        
        # Compute the gradient
        gradient = (1 / n_samples) * np.dot(X.T, errors)
        
        # Update the weights
        new_weights = weights - learning_rate * gradient
        
        # Compute the cost (Mean Squared Error)
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        cost_history.append(cost)
        
        # Check for convergence (norm of weight change below tolerance)
        if np.linalg.norm(new_weights - weights) < tolerance:
            print(f"Converged after {iteration+1} iterations")
            break
        
        # Update weights for the next iteration
        weights = new_weights
    
    return weights, cost_history

# Set parameters for gradient descent
learning_rate = 0.5  # You can adjust this based on the tuning
tolerance = 1e-6
max_iters = 10000

# Run the gradient descent algorithm
weights, cost_history = batch_gradient_descent(X_train, y_train, learning_rate, tolerance, max_iters)

# Print final weights and cost function value
print("Final Weights:", weights)
print("Final Cost:", cost_history[-1])

# Calculate the cost function value on the test data
test_cost = compute_cost(X_test, y_test, weights)
print("Cost function value on the test data:", test_cost)

# Plot the cost history to visualize convergence
import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Over Iterations')
plt.show()
