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
def entropy(data):
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

# Function to choose the best attribute based on the selected heuristic
def choose_best_attribute(data, attributes):
    base_score = entropy(data)
    best_attribute = None
    best_gain = -float('inf')
    for attribute in attributes:
        splits = split_data(data, attribute)
        weighted_score = sum((len(subset) / len(data)) * entropy(subset) for subset in splits.values())
        gain = base_score - weighted_score
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    return best_attribute

# Recursive function to build the decision tree
def id3_algorithm(data, attributes, label):
    if len(Counter(data[label])) == 1:
        return {'label': data[label].iloc[0]}
    
    if not attributes:
        majority_label = Counter(data[label]).most_common(1)[0][0]
        return {'label': majority_label}
    
    best_attribute = choose_best_attribute(data, attributes)
    tree = {best_attribute: {}}
    splits = split_data(data, best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    for attribute_value, subset in splits.items():
        if len(subset) == 0:
            majority_label = Counter(data[label]).most_common(1)[0][0]
            tree[best_attribute][attribute_value] = {'label': majority_label}
        else:
            subtree = id3_algorithm(subset, remaining_attributes, label)
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
        return find_most_common_label(tree)

def find_most_common_label(subtree):
    if 'label' in subtree:
        return subtree['label']
    labels = []
    for branch in subtree.values():
        labels.append(find_most_common_label(branch))
    return Counter(labels).most_common(1)[0][0]

# Function to calculate training error
def calculate_error(tree, data):
    incorrect_predictions = 0
    for _, instance in data.iterrows():
        if predict(tree, instance) != instance[label]:
            incorrect_predictions += 1
    return incorrect_predictions / len(data)

# Function to generate one tree (helper function for joblib)
def generate_single_tree(data, attributes, label):
    bootstrap_data = bootstrap_sample(data)
    tree = id3_algorithm(bootstrap_data, attributes, label)
    return tree

# Function to generate bagged trees in parallel
def bagged_trees_parallel(data, attributes, label, num_trees, n_jobs=12):
    trees = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_tree)(data, attributes, label)
        for _ in tqdm(range(num_trees), desc="Generating Trees")
    )
    return trees

# Function to calculate error for bagged trees in parallel
def calculate_bagging_error_parallel(trees, data, n_jobs=12):
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(predict_with_bagging)(trees, instance)
        for _, instance in tqdm(data.iterrows(), desc="Calculating Error", total=len(data))
    )
    incorrect_predictions = sum(1 for i, instance in enumerate(data.iterrows()) 
                                if predictions[i] != instance[1][label])
    return incorrect_predictions / len(data)

# Function to predict using bagged trees
def predict_with_bagging(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    majority_label = Counter(predictions).most_common(1)[0][0]
    return majority_label

# Function to calculate a bootstrap sample
def bootstrap_sample(data):
    return data.sample(frac=1, replace=True)

def main():
    # Load the datasets
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    # Add columns to the datasets
    train_data.columns = attributes + [label]
    test_data.columns = attributes + [label]
    
    # Binarize the numerical columns in both train and test data
    train_data = binarize_numerical_features(train_data, numerical_attributes)
    test_data = binarize_numerical_features(test_data, numerical_attributes)
    
    # Vary the number of trees
    num_trees_range = list(range(1, 500))
    train_errors = []
    test_errors = []

    for num_trees in num_trees_range:
        print(f"num_trees: {num_trees}")
        # Generate trees in parallel
        trees = bagged_trees_parallel(train_data, attributes, label, num_trees, n_jobs=12)
        
        # Calculate training and test errors in parallel
        train_error = calculate_bagging_error_parallel(trees, train_data, n_jobs=12)
        test_error = calculate_bagging_error_parallel(trees, test_data, n_jobs=12)
        
        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot the results
    plt.plot(num_trees_range, train_errors, label='Training Error')
    plt.plot(num_trees_range, test_errors, label='Test Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Bagging Performance')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
