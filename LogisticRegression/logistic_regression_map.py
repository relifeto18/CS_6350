import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Sigmoid function
def sigmoid(z):
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def clip_gradient(gradient, threshold=10.0):
    norm = np.linalg.norm(gradient)
    if norm > threshold:
        gradient = gradient * (threshold / norm)
    return gradient

# Objective gradient with Gaussian prior
def compute_gradient(X, y, w, v):
    predictions = sigmoid(np.dot(X, w))
    gradient = -np.dot(X.T, (y - predictions)) / X.shape[0] + (1 / max(v, 0.1)) * w
    return clip_gradient(gradient)

# Learning rate schedule
def learning_rate_schedule(gamma0, d, t):
    return gamma0 / (1 + (gamma0 / d) * t)

# Logistic regression training with SGD
def train_logistic_regression(X_train, y_train, X_test, y_test, v, gamma0, d, epochs=100):
    # Initialize weights
    w = np.zeros(X_train.shape[1])
    
    # Track errors
    train_errors = []
    test_errors = []
    update_count = 0

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        # SGD for each sample
        for i in range(X_train.shape[0]):
            # Compute gradient
            gradient = compute_gradient(X_train[i:i+1], y_train[i:i+1], w, v)

            # Update weights with learning rate schedule
            lr = learning_rate_schedule(gamma0, d, update_count)
            w -= lr * gradient

            # Increment update count
            update_count += 1

        # Compute training and test errors
        train_predictions = sigmoid(np.dot(X_train, w))
        test_predictions = sigmoid(np.dot(X_test, w))
        train_predictions = np.clip(train_predictions, 1e-15, 1 - 1e-15)
        test_predictions = np.clip(test_predictions, 1e-15, 1 - 1e-15)
        train_error = log_loss(y_train, train_predictions)
        test_error = log_loss(y_test, test_predictions)

        train_errors.append(train_error)
        test_errors.append(test_error)

    return w, train_errors, test_errors

# Main function
def main():
    # Load dataset
    train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
    test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Add bias term
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Hyperparameters
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    gamma0 = 0.5
    d = 0.5
    epochs = 100

    # Train models for each variance
    results = []
    for v in variances:
        print(f"Training for variance v = {v}")
        w, train_errors, test_errors = train_logistic_regression(X_train, y_train, X_test, y_test, v, gamma0, d, epochs)

        # Record final training and test errors
        results.append({
            "Variance": v,
            "Final Train Error": train_errors[-1],
            "Final Test Error": test_errors[-1],
        })

        # # Plot convergence curve
        # import matplotlib.pyplot as plt
        # plt.plot(train_errors, label=f"Train (v={v})")
        # plt.plot(test_errors, label=f"Test (v={v})")
        # plt.xlabel("Epochs")
        # plt.ylabel("Log Loss")
        # plt.title(f"Convergence Curve (v={v})")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    # Print summary
    print("\nSummary of Results:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
