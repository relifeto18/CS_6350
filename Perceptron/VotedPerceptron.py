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

# Initialize variables for Voted Perceptron
weights = np.zeros(X_train.shape[1])  # Includes an element for bias
epochs = 10
vote_list = []  # List to store (weight vector, count)
count = 1       # Initial count for the first weight vector

# Voted Perceptron training
for epoch in range(epochs):
    # # Shuffle the training data at the start of each epoch
    # indices = np.arange(X_train.shape[0])
    # np.random.shuffle(indices)
    # X_train, y_train = X_train[indices], y_train[indices]
    
    # Training loop
    for i in range(X_train.shape[0]):
        # Calculate the linear output
        linear_output = np.dot(X_train[i], weights)
        # Check if misclassified
        if y_train[i] * linear_output <= 0:
            # Append the current (weights, count) to vote_list
            vote_list.append((weights.copy(), count))
            # Update weights
            weights += y_train[i] * X_train[i]
            # Reset count
            count = 1
        else:
            # Increment count if correctly classified
            count += 1

# Add the last (weights, count) to the vote list
vote_list.append((weights.copy(), count))

# Evaluate on test data using the Voted Perceptron
test_predictions = np.zeros(len(X_test))
for w, c in vote_list:
    # Compute predictions with each voted weight vector
    test_predictions += c * np.sign(np.dot(X_test, w))

# Final prediction based on voting
final_predictions = np.sign(test_predictions)
test_error = np.mean(final_predictions != y_test)

# Display the results
print("List of distinct weight vectors and their counts (including bias):")
for i, (w, c) in enumerate(vote_list):
    print(f"Weight Vector {i+1}: {w[:-1]}, Bias: {w[-1]}, Count: {c}")
print("\nAverage prediction error on the test dataset:", test_error)