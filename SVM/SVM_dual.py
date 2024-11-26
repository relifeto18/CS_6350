import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load the data
train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

# Separate features and labels
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

# Convert labels to {1, -1}
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Parameters
C_values = [100 / 873, 500 / 873, 700 / 873]
n_samples, n_features = X_train.shape

# Kernel function (linear in this case)
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Compute the kernel matrix
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i, j] = linear_kernel(X_train[i], X_train[j]) * y_train[i] * y_train[j]

# Objective function for dual SVM
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

# Equality constraint: sum(alpha * y) = 0
constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_train)}

# Bounds for alpha: 0 <= alpha <= C
def train_dual_svm(C):
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Initial guess for alpha
    initial_alpha = np.zeros(n_samples)

    # Optimize using SLSQP
    result = minimize(objective, initial_alpha, bounds=bounds, constraints=constraints, method='SLSQP')
    alpha = result.x

    # Compute w from alpha
    w = np.sum(alpha[:, None] * y_train[:, None] * X_train, axis=0)

    # Find support vectors to calculate b
    support_vectors = (alpha > 1e-5) & (alpha < C - 1e-5)
    b = np.mean(y_train[support_vectors] - np.dot(X_train[support_vectors], w))

    return w, b, alpha

# Prediction function
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

# Run for each C and compare with primal results
results_dual = {}
for C in C_values:
    w_dual, b_dual, alpha_dual = train_dual_svm(C)
    results_dual[C] = {'w': w_dual, 'b': b_dual, 'alpha': alpha_dual}
    print(f"C = {C}:")
    print(f"  w (dual): {np.round(w_dual, 4)}")
    print(f"  b (dual): {b_dual:.4f}")
    
    # Training accuracy
    train_predictions = predict(X_train, w_dual, b_dual)
    train_accuracy = np.mean(train_predictions == y_train)

    # Test accuracy
    test_predictions = predict(X_test, w_dual, b_dual)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"  Training Error: {1- train_accuracy:.4f}")
    print(f"  Test Error: {1- test_accuracy:.4f}")
    print()