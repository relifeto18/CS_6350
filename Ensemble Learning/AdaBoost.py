#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Get attribute descriptions from data-desc.txt
label = 'label'

# Function to binarize numerical features based on the median
def binarize_numerical_features(data, numerical_attributes):
    for col in numerical_attributes:
        median_value = data[col].median()
        data[col] = np.where(data[col] > median_value, 1, 0)
        
    return data

# Calculate entropy for information gain
def entropy(data, weights):
    total_weight = sum(weights)
    label_weights = Counter()

    for idx, label_val in enumerate(data[label]):
        label_weights[label_val] += weights[idx]

    entropy_value = -sum((weight / total_weight) * np.log2((weight / total_weight) + 1e-10)
                         for weight in label_weights.values() if weight > 0)
    
    return entropy_value

# Function to calculate weighted information gain
def calculate_weighted_entropy(data, attribute, weights):
    splits = split_data(data, attribute)
    total_weight = sum(weights)
    
    weighted_entropy = 0
    for value, subset in splits.items():
        subset_weights = [weights[i] for i in subset.index]  # Extract weights for this subset
        if sum(subset_weights) > 0:  # Handle empty subsets
            weighted_entropy += (sum(subset_weights) / total_weight) * entropy(subset, subset_weights)
    
    return weighted_entropy

# Split the dataset based on an attribute
def split_data(data, attribute):
    return {value: data[data[attribute] == value] for value in data[attribute].unique()}

# Function to choose the best attribute based on the selected heuristic
def choose_best_attribute(data, attributes, weights):
    base_score = entropy(data, weights)
    
    best_attribute = None
    best_gain = -float('inf')
    
    for attribute in attributes:
        weighted_score = calculate_weighted_entropy(data, attribute, weights)
        gain = base_score - weighted_score
        
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
            
    return best_attribute

# Recursive function to build the decision tree
def id3_algorithm(data, attributes, label, weights, max_depth=1, depth=0):
    # Check if all labels are the same (pure node)
    if len(Counter(data[label])) == 1:
        return {'label': data[label].iloc[0]}
    
    # Check if we've reached the maximum depth
    if max_depth is not None and depth == max_depth:
        majority_label = Counter(data[label]).most_common(1)[0][0]
        return {'label': majority_label}
    
    # If there are no attributes left to split, return the majority label
    if not attributes:
        majority_label = Counter(data[label]).most_common(1)[0][0]
        return {'label': majority_label}
    
    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(data, attributes, weights)
    
    # Create a node for the best attribute
    tree = {best_attribute: {}}
    
    # Split the data based on the best attribute
    splits = split_data(data, best_attribute)
    
    # Remove the best attribute from the available attributes
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Recursively build the tree for each subset
    for attribute_value, subset in splits.items():
        subset_weights = [weights[i] for i in subset.index]
        if not subset.empty:  # Check for empty subsets
            subtree = id3_algorithm(subset, remaining_attributes, label, subset_weights, max_depth, depth + 1)
            tree[best_attribute][attribute_value] = subtree
        else:
            majority_label = Counter(data[label]).most_common(1)[0][0]
            tree[best_attribute][attribute_value] = {'label': majority_label}
        
    return tree

# Function to predict the label of a single instance
def predict(tree, instance):
    if 'label' in tree:
        return tree['label']
    
    attribute = next(iter(tree))
    attribute_value = instance[attribute]
    
    if attribute_value in tree[attribute]:
        return predict(tree[attribute][attribute_value], instance)
    else:
        # Return the most common label if the value is not found
        return None

# Function to calculate training error
def calculate_error(tree, data, weights):
    incorrect_predictions = 0
    total_weight = weights.sum()
    
    for idx, instance in data.iterrows():
        prediction = predict(tree, instance)
        if prediction != instance[label]:
            incorrect_predictions += weights[idx]
    
    error = incorrect_predictions / total_weight
    
    return error

##########################################################################################
##########################################################################################

# Function to calculate weighted error
def calculate_weighted_error(tree, data, weights):
    total_weight = sum(weights)
    incorrect_predictions = 0
    for idx, instance in data.iterrows():
        if predict(tree, instance) != instance[label]:
            incorrect_predictions += weights[idx]
    return incorrect_predictions / total_weight

# Function to make predictions using the ensemble of classifiers
def adaboost_predict(classifiers, alphas, instance):
    weighted_predictions = {}
    
    for stump, alpha in zip(classifiers, alphas):
        prediction = predict(stump, instance)
        if prediction not in weighted_predictions:
            weighted_predictions[prediction] = 0
        weighted_predictions[prediction] += alpha
    
    # Return the label with the highest weighted vote
    return max(weighted_predictions, key=weighted_predictions.get)

# Function to calculate the training error for AdaBoost
def calculate_adaboost_error(classifiers, alphas, data):
    incorrect_predictions = 0
    for _, instance in data.iterrows():
        if adaboost_predict(classifiers, alphas, instance) != instance[label]:
            incorrect_predictions += 1
    return incorrect_predictions / len(data)

# Function to make predictions using a single stump
def adaboost_predict_single_stump(stump, instance):
    return predict(stump, instance)

# AdaBoost implementation with tracking of training and test errors
def adaboost_with_error_tracking(train_data, test_data, attributes, T):
    n = len(train_data)
    weights = np.ones(n) / n  # Initialize weights equally
    
    classifiers = []
    alpha_values = []

    # Lists to store training and test errors over T iterations
    training_errors = []
    testing_errors = []
    stump_training_errors = []
    stump_testing_errors = []
    
    for t in range(T):
        print("Iteration: ", t)
        # Train a decision stump
        stump = id3_algorithm(train_data, attributes, label, weights)

        # Calculate the weighted error of the stump
        error = calculate_weighted_error(stump, train_data, weights)
        
        if error == 0:  # Perfect classification
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - error) / error)
        
        # Store classifier and its alpha value
        classifiers.append(stump)
        alpha_values.append(alpha)
        
        # Update weights
        for i in range(n):
            prediction = adaboost_predict_single_stump(stump, train_data.iloc[i])
            if prediction == train_data.iloc[i][label]:
                weights[i] *= np.exp(-alpha)
            else:
                weights[i] *= np.exp(alpha)
                
        # Normalize weights
        weights /= sum(weights)
        
        # Track training and test errors after each iteration for the ensemble
        training_error = calculate_adaboost_error(classifiers, alpha_values, train_data)
        testing_error = calculate_adaboost_error(classifiers, alpha_values, test_data)
        training_errors.append(training_error)
        testing_errors.append(testing_error)

        # Track training and test errors for the current stump (weak classifier)
        stump_training_error = calculate_error(stump, train_data, weights)
        stump_testing_error = calculate_error(stump, test_data, weights)
        stump_training_errors.append(stump_training_error)
        stump_testing_errors.append(stump_testing_error)

    return classifiers, alpha_values, training_errors, testing_errors, stump_training_errors, stump_testing_errors

##########################################################################################
##########################################################################################

def main():
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 
                'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # Load the datasets
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    # Add one row in the datasets
    train_data.columns = attributes + [label]
    test_data.columns = attributes + [label]
    
    # Binarize the numerical columns in both train and test data
    train_data = binarize_numerical_features(train_data, numerical_attributes)
    test_data = binarize_numerical_features(test_data, numerical_attributes)
    
    T = 500
    
    # Run AdaBoost on the training data
    classifiers, alphas, training_errors, testing_errors, stump_training_errors, stump_testing_errors = adaboost_with_error_tracking(
        train_data, test_data, attributes, T
    )
            
    # Plotting training and test errors over iterations T
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, T + 1), training_errors, label='Training Error')
    plt.plot(range(1, T + 1), testing_errors, label='Test Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error')
    plt.title('Training and Test Errors Over Iterations (T)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting training and test errors of individual decision stumps
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, T + 1), stump_training_errors, label='Stump Training Error')
    plt.plot(range(1, T + 1), stump_testing_errors, label='Stump Test Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Stump Error')
    plt.title('Training and Test Errors of Decision Stumps')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Run AdaBoost on the training data
    # classifiers, alphas = adaboost(train_data, attributes, T)

    # # Calculate training and testing errors
    # training_error = calculate_adaboost_error(classifiers, alphas, train_data)
    # testing_error = calculate_adaboost_error(classifiers, alphas, test_data)
    
    # print(f"Training Error: {training_error:.4f}")
    # print(f"Testing Error: {testing_error:.4f}\n")


if __name__ == '__main__':
    main()