import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Learning rate schedule
def learning_rate_schedule(gamma0, d, t):
    return gamma0 / (1 + gamma0 * d * t)

# Logistic Regression using Maximum Likelihood (ML)
def logistic_regression_ml(X_train, y_train, X_test, y_test, gamma0, d, num_epochs, variance):
    n_samples, n_features = X_train.shape
    # Initialize weights with a Gaussian distribution, variance `v`
    # np.random.seed(42)
    weights = np.random.normal(0, np.sqrt(variance), size=n_features)
    # np.random.seed(None)
    train_errors = []
    test_errors = []

    for epoch in range(num_epochs):
        # Shuffle training data at the start of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(n_samples):
            # Compute learning rate for the current step
            gamma = learning_rate_schedule(gamma0, d, epoch * n_samples + i)
            
            # Calculate the prediction (sigmoid of linear combination)
            xi = X_train[i]
            yi = y_train[i]
            prediction = sigmoid(np.dot(weights, xi))
            
            # Gradient of the negative log-likelihood
            gradient = (prediction - yi) * xi
            
            # Update weights
            weights -= gamma * gradient

        # Compute training and test errors (negative log-likelihood)
        train_preds = sigmoid(np.dot(X_train, weights))
        test_preds = sigmoid(np.dot(X_test, weights))
        train_error = log_loss(y_train, train_preds)
        test_error = log_loss(y_test, test_preds)
        train_errors.append(train_error)
        test_errors.append(test_error)

    return weights, train_errors, test_errors

# Load training and testing data
train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

# Extract features and labels
X_train = train_data.iloc[:, :-1].values  # Features
y_train = train_data.iloc[:, -1].values   # Labels
X_test = test_data.iloc[:, :-1].values    # Features
y_test = test_data.iloc[:, -1].values     # Labels

# Hyperparameters for learning rate schedule
gamma0 = 0.1  # Initial learning rate
d = 0.01      # Decay factor
num_epochs = 100  # Number of epochs
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]  # Variance values

# Train the logistic regression model for each variance
results = []
for variance in variances:
    weights, train_errors, test_errors = logistic_regression_ml(
        X_train, y_train, X_test, y_test, gamma0, d, num_epochs, variance
    )
    results.append((variance, train_errors[-1], test_errors[-1]))

# Display results
results_df = pd.DataFrame(results, columns=["Variance", "Final Train Error", "Final Test Error"])
print(results_df)

# Plot convergence for one example variance (e.g., the first one)
variance_example = variances[0]
_, train_errors, test_errors = logistic_regression_ml(X_train, y_train, X_test, y_test, gamma0, d, num_epochs, variance_example)

# # Plot the convergence curve
# plt.figure(figsize=(10, 6))
# epochs = range(1, len(train_errors) + 1)
# plt.plot(epochs, train_errors, label='Training Error', marker='o')
# plt.plot(epochs, test_errors, label='Test Error', marker='s')
# plt.title(f'Convergence Curve (Variance = {variance_example})', fontsize=16)
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Log-Loss', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.show()
