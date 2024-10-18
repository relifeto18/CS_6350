#!/usr/bin/env python

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Get attribute descriptions from data-desc.txt
label = 'label'
attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 
                'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Function to binarize numerical features based on the median
def binarize_numerical_features(data, numerical_attributes):
    for col in numerical_attributes:
        median_value = data[col].median()
        data[col] = np.where(data[col] > median_value, 1, 0)
        
    return data

# Calculate entropy for information gain
def entropy(data, label):
    labels = data[label]
    label_counts = Counter(labels)
    total = len(labels)
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values())
    
    return entropy_value

# Split the dataset based on an attribute
def split_data(data, attribute):
    values = data[attribute].unique()
    subset = {value: data[data[attribute] == value] for value in values}
    
    return subset

# Randomly select a subset of features before each split
def choose_best_attribute_random_subset(data, attributes, subset_size, label):
    # Randomly select a subset of features
    selected_attributes = random.sample(attributes, subset_size)
    
    base_score = entropy(data, label)
    best_attribute = None
    best_gain = -float('inf')
    
    for attribute in selected_attributes:
        splits = split_data(data, attribute)
        weighted_score = sum((len(subset) / len(data)) * entropy(subset, label) for subset in splits.values())
        gain = base_score - weighted_score
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    return best_attribute, best_gain

# Recursive function to build the decision tree
def id3_algorithm(data, attributes, subset_size, label):
    # Check if all labels are the same (pure node)
    if len(Counter(data[label])) == 1:
        return {'label': data[label].iloc[0]}
    
    # # If there are no attributes left to split, return the majority label
    # if not attributes or len(data) == 0:
    #     majority_label = Counter(data[label]).most_common(1)[0][0]
    #     return {'label': majority_label}
    
    # Choose the best attribute to split on
    best_attribute, best_gain = choose_best_attribute_random_subset(data, attributes, subset_size, label)
    
    if best_gain == 0:
        majority_label = Counter(data[label]).most_common(1)[0][0]
        return {'label': majority_label}
    
    # Create a node for the best attribute
    tree = {best_attribute: {}}
    
    # Split the data based on the best attribute
    splits = split_data(data, best_attribute)
    
    # Recursively build the tree for each subset
    for attribute_value, subset in splits.items():
        if len(subset) == 0:
            majority_label = Counter(data[label]).most_common(1)[0][0]
            tree[best_attribute][attribute_value] = {'label': majority_label}
        else:
            subtree = id3_algorithm(subset, attributes, subset_size, label)
            tree[best_attribute][attribute_value] = subtree
    
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
        # return 'no'
        return Counter([predict(subtree, instance) for subtree in tree[attribute].values()]).most_common(1)[0][0]

#####################################################################################
#####################################################################################

# Function to calculate error
def calculate_error(trees, data):
    incorrect_predictions = 0
    for _, instance in data.iterrows():
        predictions = [predict(tree, instance) for tree in trees]
        majority_label = Counter(predictions).most_common(1)[0][0]  # Majority voting
        if majority_label != instance['label']:
            incorrect_predictions += 1
    return incorrect_predictions / len(data)

# Function to generate a bootstrapped sample from the training data
def bootstrap_sample(data):
    n = len(data)
    sample_indices = [random.randint(0, n-1) for _ in range(n)]
    return data.iloc[sample_indices]

# Function to train random forest
def random_forest_algorithm(train_data, test_data, num_trees, subset_size, label, attributes):
    trees = []
    training_errors = []
    testing_errors = []
    
    for i in range(1, num_trees + 1):
        print("subset_size, num_trees: ", subset_size, i)
        # Generate a bootstrapped sample
        bootstrapped_data = bootstrap_sample(train_data)
        
        # Train a decision tree on the bootstrapped sample
        tree = id3_algorithm(bootstrapped_data, attributes, subset_size, label)
        trees.append(tree)
        
        # Calculate training and testing error
        train_error = calculate_error(trees, train_data)
        test_error = calculate_error(trees, test_data)
        
        training_errors.append(train_error)
        testing_errors.append(test_error)
    
    return training_errors, testing_errors

# Function to train random forest with parallelism and progress bar
def parallel_random_forest_algorithm(train_data, test_data, num_trees, subset_size, label, attributes, n_jobs=-1):
    # Helper function to train a single tree
    def train_single_tree(i):
        # Generate a bootstrapped sample
        bootstrapped_data = bootstrap_sample(train_data)
        
        # Train a decision tree on the bootstrapped sample
        tree = id3_algorithm(bootstrapped_data, attributes, subset_size, label)
        return tree

    # Train trees in parallel with tqdm progress bar
    trees = Parallel(n_jobs=n_jobs)(
        delayed(train_single_tree)(i) for i in tqdm(range(1, num_trees + 1), desc=f"Training {num_trees} Trees")
    )

    # After training, calculate training and testing errors
    training_errors = []
    testing_errors = []

    for i in range(1, num_trees + 1):
        # Calculate training and testing error for the first `i` trees
        train_error = calculate_error(trees[:i], train_data)
        test_error = calculate_error(trees[:i], test_data)
        
        training_errors.append(train_error)
        testing_errors.append(test_error)
    
    return training_errors, testing_errors
#####################################################################################
#####################################################################################

def main():
    # Load the datasets
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    # Add one row in the datasets
    train_data.columns = attributes + [label]
    test_data.columns = attributes + [label]
    
    # Binarize the numerical columns in both train and test data
    train_data = binarize_numerical_features(train_data, numerical_attributes)
    test_data = binarize_numerical_features(test_data, numerical_attributes)
    
    # Experiment with varying number of trees and feature subsets
    num_trees = 500
    feature_subset_sizes = [2, 4, 6]
    
    plt.figure(figsize=(12, 8))
    
    for subset_size in feature_subset_sizes:
        print("subset_size: ", subset_size)
        training_errors = []
        testing_errors = []
        
        train_err, test_err = parallel_random_forest_algorithm(train_data, test_data, num_trees, subset_size, label, attributes, n_jobs=10)
        training_errors.append(train_err[-1])
        testing_errors.append(test_err[-1])
        
        # Plot errors for each subset size
        plt.plot(range(1, num_trees + 1), test_err, label=f'Test Error (Subset Size = {subset_size})')
        plt.plot(range(1, num_trees + 1), train_err, label=f'Train Error (Subset Size = {subset_size})', linestyle='--')
    
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Random Forest Training and Testing Errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()