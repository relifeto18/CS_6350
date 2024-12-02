import argparse
import numpy as np
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Neural Network Gradient Computation")
parser.add_argument("--hidden_layer_1_size", type=int, default=2, help="Number of neurons in the first hidden layer")
parser.add_argument("--hidden_layer_2_size", type=int, default=2, help="Number of neurons in the second hidden layer")

args = parser.parse_args()
hidden_layer_1_size = args.hidden_layer_1_size
hidden_layer_2_size = args.hidden_layer_2_size

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Load data
train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

# Neural network parameters
input_size = X_train.shape[1]
output_size = 1

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size + 1, hidden_layer_1_size)
W2 = np.random.randn(hidden_layer_1_size + 1, hidden_layer_2_size)
W3 = np.random.randn(hidden_layer_2_size + 1, output_size)

# Add bias to input
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

for i in range(X_train.shape[0]):
    # Forward pass
    a0 = X_train[i].reshape(1, -1)  # Input layer (with bias)
    y_true = y_train[i]

    z1 = np.dot(a0, W1)
    a1 = np.hstack([np.ones((1, 1)), sigmoid(z1)])  # Hidden Layer 1 (with bias)
    z2 = np.dot(a1, W2)
    a2 = np.hstack([np.ones((1, 1)), sigmoid(z2)])  # Hidden Layer 2 (with bias)
    z3 = np.dot(a2, W3)
    y_pred = z3  # Output layer (no activation)

    # Backpropagation
    delta3 = y_pred - y_true  # Error at output layer
    grad_W3 = np.dot(a2.T, delta3)

    delta2 = np.dot(delta3, W3[1:].T) * sigmoid_derivative(z2)
    grad_W2 = np.dot(a1.T, delta2)

    delta1 = np.dot(delta2, W2[1:].T) * sigmoid_derivative(z1)
    grad_W1 = np.dot(a0.T, delta1)

# Output the gradients
print()
print("Gradient for W1 (Input to Hidden Layer 1):")
print(grad_W1)
print("\nGradient for W2 (Hidden Layer 1 to Hidden Layer 2):")
print(grad_W2)
print("\nGradient for W3 (Hidden Layer 2 to Output):")
print(grad_W3)
    