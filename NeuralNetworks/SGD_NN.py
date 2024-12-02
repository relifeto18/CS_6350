import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Learning rate schedule
def learning_rate_schedule(gamma0, d, t):
    return gamma0 / (1 + (gamma0 / d) * t)

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to compute forward pass and loss
def forward_pass(X, W1, W2, W3):
    z1 = np.dot(X, W1)
    a1 = np.hstack([np.ones((X.shape[0], 1)), sigmoid(z1)])  # Add bias
    z2 = np.dot(a1, W2)
    a2 = np.hstack([np.ones((X.shape[0], 1)), sigmoid(z2)])  # Add bias
    z3 = np.dot(a2, W3)
    y_pred = z3  # No activation in the output layer
    return a1, a2, y_pred

# Function to compute gradients using backpropagation
def compute_gradients(X, y_true, a1, a2, y_pred, W2, W3):
    delta3 = y_pred - y_true  # Error at output layer
    grad_W3 = np.dot(a2.T, delta3)

    delta2 = np.dot(delta3, W3[1:].T) * sigmoid_derivative(a2[:, 1:])
    grad_W2 = np.dot(a1.T, delta2)

    delta1 = np.dot(delta2, W2[1:].T) * sigmoid_derivative(a1[:, 1:])
    grad_W1 = np.dot(X.T, delta1)

    return grad_W1, grad_W2, grad_W3

# Main function to evaluate different hidden layer widths
def main():
    # Load data
    train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
    test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

    # Add bias to input data
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Parameters
    hidden_layer_widths = [5, 10, 25, 50, 100]  # Choices for hidden layer width
    gamma0 = 0.1  # Initial learning rate
    d = 0.95  # Decay factor
    epochs = 100
    batch_size = 1

    results = {}

    # Loop over hidden layer widths
    # plt.figure()
    # plt.autoscale()
    for hidden_layer_width in hidden_layer_widths:
        print(f"\nEvaluating for hidden_layer_width = {hidden_layer_width}")

        # Initialize weights with standard Gaussian distribution
        input_size = X_train.shape[1]
        output_size = 1
        
        np.random.seed(11)
        W1 = np.random.randn(input_size, hidden_layer_width)
        W2 = np.random.randn(hidden_layer_width + 1, hidden_layer_width)
        W3 = np.random.randn(hidden_layer_width + 1, output_size)
        np.random.seed(None)  # Remove the seed
        
        # W1 = np.zeros((input_size, hidden_layer_width))
        # W2 = np.zeros((hidden_layer_width + 1, hidden_layer_width))
        # W3 = np.zeros((hidden_layer_width + 1, output_size))

        # Variables to track convergence
        objective_curve = []
        update_count = 0

        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Perform SGD
            for t in range(X_train.shape[0] // batch_size):
                start = t * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Forward pass
                a1, a2, y_pred = forward_pass(X_batch, W1, W2, W3)

                # Compute gradients
                grad_W1, grad_W2, grad_W3 = compute_gradients(X_batch, y_batch, a1, a2, y_pred, W2, W3)

                # Update weights using learning rate schedule
                lr = learning_rate_schedule(gamma0, d, update_count)
                W1 -= lr * grad_W1
                W2 -= lr * grad_W2
                W3 -= lr * grad_W3

                # Track objective function (training error)
                _, _, train_preds = forward_pass(X_train, W1, W2, W3)
                train_error = mean_squared_error(y_train, train_preds)
                objective_curve.append(train_error)

                # Increment update count
                update_count += 1

            # Compute test error at the end of each epoch
            _, _, test_preds = forward_pass(X_test, W1, W2, W3)
            test_error = mean_squared_error(y_test, test_preds)
            # print(f"Epoch {epoch + 1}/{epochs} - Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

        # Store results for this hidden layer width
        results[hidden_layer_width] = (objective_curve, train_error, test_error)

        # Plot convergence curve for this width
    #     plt.plot(objective_curve, label=f"Width {hidden_layer_width}")

    # # Finalize and show the combined plot
    # plt.xlabel("Number of Updates")
    # plt.ylabel("Objective Function (MSE)")
    # plt.title("Convergence Diagnosis for Different Hidden Layer Widths")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Print summary results
    print("\nSummary of Results:")
    for width, (curve, train_err, test_err) in results.items():
        print(f"Width {width} - Final Training Error: {train_err:.4f}, Final Test Error: {test_err:.4f}")

if __name__ == "__main__":
    main()