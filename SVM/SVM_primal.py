import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
C_values = [100 / 873, 500 / 873, 700 / 873]  # Different values of C
T = 100  # Maximum epochs
gamma_0 = 0.1  # Initial learning rate parameter (to be tuned)
a = 0.1  # Learning rate decay parameter (to be tuned)
np.random.seed(12)

# Define the learning rate schedule
def learning_rate_1(t):
    return gamma_0 / (1 + gamma_0 * t / a)

def learning_rate_2(t):
    return gamma_0 / (1 + t)

# Initialize weight vector and bias
def initialize_weights(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b

# Objective function
def objective_function(w, b, X, y, C):
    hinge_loss = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    return 0.5 * np.dot(w, w) + C * np.sum(hinge_loss)

# Train SVM with SGD
def train_svm_sgd_1(X, y, C):
    n_samples, n_features = X.shape
    w, b = initialize_weights(n_features)
    objective_values = []

    for epoch in range(T):
        # Shuffle the training examples
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for i in range(n_samples):
            t = epoch * n_samples + i  # Update count
            eta_t = learning_rate_1(t)  # Learning rate

            # Compute sub-gradient for hinge loss
            if y[i] * (np.dot(w, X[i]) + b) < 1:
                w = (1 - eta_t) * w + eta_t * C * y[i] * X[i]
                b += eta_t * C * y[i]
            else:
                w = (1 - eta_t) * w

            # Calculate and record the objective function value
            if i % 100 == 0:  # Sample objective calculation periodically
                obj_value = objective_function(w, b, X, y, C)
                objective_values.append(obj_value)

    return w, b, objective_values

def train_svm_sgd_2(X, y, C):
    n_samples, n_features = X.shape
    w, b = initialize_weights(n_features)
    objective_values = []

    for epoch in range(T):
        # Shuffle the training examples
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for i in range(n_samples):
            t = epoch * n_samples + i  # Update count
            eta_t = learning_rate_2(t)  # Learning rate

            # Compute sub-gradient for hinge loss
            if y[i] * (np.dot(w, X[i]) + b) < 1:
                w = (1 - eta_t) * w + eta_t * C * y[i] * X[i]
                b += eta_t * C * y[i]
            else:
                w = (1 - eta_t) * w

            # Calculate and record the objective function value
            if i % 100 == 0:  # Sample objective calculation periodically
                obj_value = objective_function(w, b, X, y, C)
                objective_values.append(obj_value)

    return w, b, objective_values

# Test the SVM
def test_svm(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    accuracy = np.mean(predictions == y)
    return accuracy

# Run Schedule 1 for each value of C
results_schedule_1 = {}
print("Results for Schedule 1 (gamma_t = gamma_0 / (1 + gamma_0 * t / a)):")
for C in C_values:
    w, b, objective_values = train_svm_sgd_1(X_train, y_train, C)
    train_accuracy = test_svm(X_train, y_train, w, b)
    test_accuracy = test_svm(X_test, y_test, w, b)
    results_schedule_1[C] = {
        'w': w, 'b': b,
        'train_error': 1 - train_accuracy,
        'test_error': 1 - test_accuracy,
        'objective_values': objective_values
    }
    
    # Print results for Schedule 1
    print(f'C={C}:')
    print(f'  weights: {np.round(w, 4)}, bias: {b:.4f}')
    print(f'  Train Error: {1 - train_accuracy:.4f}, Test Error: {1 - test_accuracy:.4f}')

# Run Schedule 2 for each value of C
results_schedule_2 = {}
print()
print("Results for Schedule 2 (gamma_t = gamma_0 / (1 + t)):")
for C in C_values:
    w, b, objective_values = train_svm_sgd_2(X_train, y_train, C)
    train_accuracy = test_svm(X_train, y_train, w, b)
    test_accuracy = test_svm(X_test, y_test, w, b)
    results_schedule_2[C] = {
        'w': w, 'b': b,
        'train_error': 1 - train_accuracy,
        'test_error': 1 - test_accuracy,
        'objective_values': objective_values
    }
    
    # Print results for Schedule 2
    print(f'C={C}:')
    print(f'  weights: {np.round(w, 4)}, bias: {b:.4f}')
    print(f'  Train Error: {1 - train_accuracy:.4f}, Test Error: {1 - test_accuracy:.4f}')
    
# Comparison between the two schedules for each C
print()
print("Comparison between Schedule 1 and Schedule 2:")
for C in C_values:
    w1, b1 = results_schedule_1[C]['w'], results_schedule_1[C]['b']
    w2, b2 = results_schedule_2[C]['w'], results_schedule_2[C]['b']
    
    # train_error_diff = abs(results_schedule_1[C]['train_error'] - results_schedule_2[C]['train_error'])
    # test_error_diff = abs(results_schedule_1[C]['test_error'] - results_schedule_2[C]['test_error'])
    # weight_diff = np.linalg.norm(w1 - w2)
    # bias_diff = abs(b1 - b2)
    train_error_diff = results_schedule_1[C]['train_error'] - results_schedule_2[C]['train_error']
    test_error_diff = results_schedule_1[C]['test_error'] - results_schedule_2[C]['test_error']
    weight_diff = w1 - w2
    bias_diff = b1 - b2
    
    print(f'C={C}:')
    # print(f'  Weight difference: {weight_diff:.4f}')
    print(f'  Weight difference: {np.round(weight_diff, 4)}')
    print(f'  Bias difference: {bias_diff:.4f}')
    print(f'  Training error difference: {train_error_diff:.4f}')
    print(f'  Test error difference: {test_error_diff:.4f}')
    print()

# Plot the objective function convergence for both schedules
for C in C_values:
    plt.plot(results_schedule_1[C]['objective_values'], label=f'Schedule 1, C={C}')
    plt.plot(results_schedule_2[C]['objective_values'], label=f'Schedule 2, C={C}', linestyle='--')

plt.xlabel('Number of updates')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Convergence for Both Schedules')
plt.legend()
plt.show()