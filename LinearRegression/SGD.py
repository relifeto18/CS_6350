import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('./data/concrete/concrete/train.csv')
X_train = train_data.iloc[:, :-1].values  # Extract features (first 7 columns)
y_train = train_data.iloc[:, -1].values  # Extract target (last column)

# Add a bias term (column of ones) to the features matrix
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term

# Define the stochastic gradient descent function
def stochastic_gradient_descent(X, y, learning_rate=0.01, tolerance=1e-6, max_updates=10000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to zeros
    cost_history = []
    
    for update in range(max_updates):
        # Randomly sample a training example
        i = np.random.randint(n_samples)
        xi = X[i, :].reshape(1, -1)
        yi = y[i]
        
        # Compute the prediction
        prediction = np.dot(xi, weights)
        
        # Compute the error
        error = prediction - yi
        
        # Compute the gradient for the current example
        gradient = xi.T * error
        
        # Update the weights
        weights = weights - learning_rate * gradient.flatten()
        
        # Calculate the cost function (MSE) for the entire training data
        predictions = np.dot(X, weights)
        errors = predictions - y
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        cost_history.append(cost)
        
        # Check if the norm is below the tolerance (convergence criterion)
        if update > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged after {update+1} updates")
            break
    
    return weights, cost_history

# Set parameters for stochastic gradient descent
learning_rate = 0.005  # You can adjust this value
tolerance = 1e-6
max_updates = 100000

# Run the stochastic gradient descent algorithm
weights, cost_history = stochastic_gradient_descent(X_train, y_train, learning_rate, tolerance, max_updates)

# Print final weights and plot the cost function values over updates
print("Final Weights:", weights)
print("Final Training Cost:", cost_history[-1])

# Plot the cost history for the training data
plt.plot(cost_history)
plt.xlabel('Updates')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Over Updates (SGD)')
plt.show()

# Now load the test data and compute cost using the final weights
test_data = pd.read_csv('./data/concrete/concrete/test.csv')
X_test = test_data.iloc[:, :-1].values  # Extract features (first 7 columns)
y_test = test_data.iloc[:, -1].values  # Extract target (last column)

# Add a bias term (column of ones) to the features matrix
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Add intercept term

# Calculate the cost function value on the test data
def compute_cost(X, y, weights):
    n_samples = len(y)
    predictions = np.dot(X, weights)
    errors = predictions - y
    cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
    return cost

test_cost = compute_cost(X_test, y_test, weights)
print("Cost function value on the test data:", test_cost)
