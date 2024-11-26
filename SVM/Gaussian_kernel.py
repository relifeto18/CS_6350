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
gamma_values = [0.1, 0.5, 1, 5, 100]
n_samples, n_features = X_train.shape

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

# Compute the kernel matrix for a given gamma
def compute_kernel_matrix(X, y, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma) * y[i] * y[j]
    return K

# Objective function for dual SVM
def objective(alpha, K):
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

# Equality constraint: sum(alpha * y) = 0
constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_train)}

# Bounds for alpha: 0 <= alpha <= C
def train_dual_svm(C, gamma):
    K = compute_kernel_matrix(X_train, y_train, gamma)
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Initial guess for alpha
    initial_alpha = np.zeros(n_samples)

    # Optimize using SLSQP
    result = minimize(objective, initial_alpha, args=(K,), bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-9})
    alpha = result.x

    # Find support vectors to calculate bias b
    support_vectors = (alpha > 1e-5) & (alpha < C - 1e-5)
    b = np.mean(
        y_train[support_vectors] - np.sum(alpha * y_train * K[support_vectors], axis=1)
    )

    return alpha, b

# Prediction function using Gaussian kernel
def predict(X, alpha, b, gamma):
    n_samples_train = X_train.shape[0]
    K_test = np.zeros((X.shape[0], n_samples_train))
    for i in range(X.shape[0]):
        for j in range(n_samples_train):
            K_test[i, j] = gaussian_kernel(X[i], X_train[j], gamma)
    return np.sign(np.dot(K_test, alpha * y_train) + b)

# Evaluate the model
results = {}
previous_support_indices = None

for C in C_values:
    for gamma in gamma_values:
        # Train the model
        alpha, b = train_dual_svm(C, gamma)

        # Training accuracy
        train_predictions = predict(X_train, alpha, b, gamma)
        train_accuracy = np.mean(train_predictions == y_train)

        # Test accuracy
        test_predictions = predict(X_test, alpha, b, gamma)
        test_accuracy = np.mean(test_predictions == y_test)
        
        # Identify support vectors
        support_indices = (alpha > 1e-5)
        num_support_vectors = np.sum(support_indices)

        # Store the results
        results[(C, gamma)] = {
            'training_error': 1 - train_accuracy,
            'test_error': 1 - test_accuracy,
        }

        # Print results for each combination
        print(f"C = {C}, gamma = {gamma}:")
        print(f"  Training Error: {1 - train_accuracy:.4f}")
        print(f"  Test Error: {1 - test_accuracy:.4f}")
        print(f"  Number of Support Vectors = {num_support_vectors}")
        print()
        
        # For C = 500/873, calculate overlap of support vectors
        if C == 500 / 873:
            if previous_support_indices is not None:
                overlap = np.sum(support_indices & previous_support_indices)
                print(f"  Overlap with Previous Gamma = {overlap}")
            previous_support_indices = support_indices

# Identify the best combination of C and gamma
best_combination = min(results, key=lambda k: results[k]['test_error'])
print("Best combination:")
print(f"C = {best_combination[0]}, gamma = {best_combination[1]}")
print(f"Training Error: {results[best_combination]['training_error']:.4f}")
print(f"Test Error: {results[best_combination]['test_error']:.4f}")