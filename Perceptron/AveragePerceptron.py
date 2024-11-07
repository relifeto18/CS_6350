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

# Add a bias term (column of ones) to X_train and X_test
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

# Initialize weights and their averages, including bias as part of weights
weights = np.zeros(X_train.shape[1])   # Last element represents bias
avg_weights = np.zeros(X_train.shape[1])
epochs = 10
weight_vectors = []  # List to store weight vectors encountered

# Average Perceptron training
for epoch in range(epochs):
    # # Shuffle the training data at the start of each epoch
    # indices = np.arange(X_train.shape[0])
    # np.random.shuffle(indices)
    # X_train, y_train = X_train[indices], y_train[indices]
    
    # Training loop
    for i in range(X_train.shape[0]):
        # Calculate the linear output, treating bias as part of weights
        linear_output = np.dot(X_train[i], weights)
        # Perceptron update rule
        if y_train[i] * linear_output <= 0:
            weight_vectors.append(weights.copy())
            weights += y_train[i] * X_train[i]
        
        # Accumulate weights for averaging
        avg_weights += weights

# Append the final weight vector
weight_vectors.append(weights.copy())

# # Final averaged weights
# avg_weights /= (epochs * X_train.shape[0])

# Evaluate on test data using the averaged weights
predictions = np.sign(np.dot(X_test, avg_weights))
test_error = np.mean(predictions != y_test)

# Print the list of weight vectors encountered (including bias as last element)
print("\nList of weight vectors encountered during training (including bias):")
for idx, vec in enumerate(weight_vectors):
    print(f"Update {idx+1}: {vec}")
    
# Display the results
print("Learned averaged weight vector (including bias):", avg_weights)
print("Average prediction error on the test dataset:", test_error)
