#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter

# Get attribute descriptions from data-desc.txt
label = 'label'
attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 
                'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
columns_with_unknown = ['job', 'education', 'contact', 'poutcome']

##########################################################################################
##########################################################################################

# Function to binarize numerical features based on the median
def binarize_numerical_features(data, numerical_attributes):
    for col in numerical_attributes:
        median_value = data[col].median()
        data[col] = np.where(data[col] > median_value, 1, 0)
        
    return data

# Function to replace "unknown" with the majority value in the training set
def replace_unknown_with_majority(data, columns_with_unknown):
    for col in columns_with_unknown:
        valid_rows = data[data[col] != "unknown"]
        valid_values = valid_rows[col] 
        majority_value = Counter(valid_values).most_common(1)[0][0]
        
        data[col] = data[col].replace("unknown", majority_value)
        
    return data

##########################################################################################
##########################################################################################

# Calculate entropy for information gain
def entropy(data):
    labels = data[label]
    label_counts = Counter(labels)
    total = len(labels)
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values())
    
    return entropy_value

# Calculate majority error
def majority_error(data):
    labels = data[label]
    label_counts = Counter(labels)
    most_common_label_count = label_counts.most_common(1)[0][1]
    total = len(labels)
    majorityerror = 1 - (most_common_label_count / total)
    
    return majorityerror

# Calculate Gini index
def gini_index(data):
    labels = data[label]
    label_counts = Counter(labels)
    total = len(labels)
    gini_value = 1 - sum((count / total) ** 2 for count in label_counts.values())
    
    return gini_value

# # Function to calculate information gain
# def information_gain(data, attribute):
#     base_entropy = entropy(data)
#     splits = split_data(data, attribute)
    
#     # Weighted entropy after the split
#     weighted_entropy = sum((len(subset) / len(data)) * entropy(subset) for subset in splits.values())
#     gain = base_entropy - weighted_entropy
    
#     return gain

# Split the dataset based on an attribute
def split_data(data, attribute):
    values = data[attribute].unique()
    subset = {value: data[data[attribute] == value] for value in values}
    
    return subset

# Function to choose the best attribute based on the selected heuristic
def choose_best_attribute(data, attributes, heuristic='entropy'):
    if heuristic == 'entropy':
        base_score = entropy(data)
    elif heuristic == 'majority_error':
        base_score = majority_error(data)
    else:
        base_score = gini_index(data)
    
    # Calculate information gain or reduction in Gini index or majority error for each attribute
    best_attribute = None
    best_gain = -float('inf')
    
    for attribute in attributes:
        splits = split_data(data, attribute)
        weighted_score = sum((len(subset) / len(data)) * (entropy(subset) if heuristic == 'entropy'
                                                          else majority_error(subset) if heuristic == 'majority_error'
                                                          else gini_index(subset))
                            for subset in splits.values())
        
        gain = base_score - weighted_score
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    return best_attribute

# Recursive function to build the decision tree
def id3_algorithm(data, attributes, label, max_depth=None, depth=0, heuristic='entropy'):
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
    best_attribute = choose_best_attribute(data, attributes, heuristic)
    
    # Create a node for the best attribute
    tree = {best_attribute: {}}
    
    # Split the data based on the best attribute
    splits = split_data(data, best_attribute)
    
    # Remove the best attribute from the available attributes
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Recursively build the tree for each subset
    for attribute_value, subset in splits.items():
        subtree = id3_algorithm(subset, remaining_attributes, label, max_depth, depth + 1, heuristic)
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
        return None

# Function to calculate training error
def calculate_error(tree, data):
    incorrect_predictions = 0
    for _, instance in data.iterrows():
        if predict(tree, instance) != instance[label]:
            incorrect_predictions += 1
    return incorrect_predictions / len(data)


def main():
    # Load the datasets
    train_data = pd.read_csv('./data/bank/train.csv', header=None)
    test_data = pd.read_csv('./data/bank/test.csv', header=None)

    # Add one row in the datasets
    train_data.columns = attributes + [label]
    test_data.columns = attributes + [label]

    # Get user input for unknown decision and max_depth
    treat_unknown = int(input("Treat unknown value as a particular attribute value (0) or as missing attribute value (1): ").strip())
    max_depth = int(input("Set the maximum tree depth (1-16): ").strip())
    
    if max_depth == '':
        max_depth = 16  # Default to be maximum
    else:
        max_depth = int(max_depth)
        
    if treat_unknown == '':
        treat_unknown = 1  # Default to treat as missing attribute value
    else:
        treat_unknown = int(treat_unknown)

    if treat_unknown:
        # Replace "unknown" values with the majority value in the training set
        train_data = replace_unknown_with_majority(train_data, columns_with_unknown)
        test_data = replace_unknown_with_majority(test_data, columns_with_unknown)
    
    # Binarize the numerical columns in both train and test data
    train_data = binarize_numerical_features(train_data, numerical_attributes)
    test_data = binarize_numerical_features(test_data, numerical_attributes)
    
    heuristics = ['entropy', 'majority_error', 'gini']
    
    print(f"Results for max depth: {max_depth}\n")
    # Loop through all three heuristics
    for heuristic in heuristics:
        print(f"Running ID3 algorithm with heuristic: {heuristic}")
        
        # for depth in range(1, max_depth+1):
        #     print(f"Building tree with max depth: {depth}")
        
        # Build a decision tree using the ID3 algorithm
        tree = id3_algorithm(train_data, attributes, label, max_depth=max_depth, heuristic=heuristic)
        
        # Calculate training error
        training_error = calculate_error(tree, train_data)
        testing_error = calculate_error(tree, test_data)
        
        print(f"Training Error: {training_error:.4f}")
        print(f"Testing Error: {testing_error:.4f}\n")


if __name__ == '__main__':
    main()