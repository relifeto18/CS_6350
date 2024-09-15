#!/usr/bin/env python
 
import pandas as pd
import numpy as np
from collections import Counter

# Load the datasets (adjust the paths accordingly)
train_data = pd.read_csv('./data/car/train.csv', header=None)
test_data = pd.read_csv('./data/car/test.csv', header=None)

# Define attributes and label column names
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
label = 'label'

# Set the columns for train and test data
train_data.columns = attributes + [label]
test_data.columns = attributes + [label]

# Entropy function
def entropy(data):
    labels = data[label]
    label_counts = Counter(labels)
    total = len(labels)
    
    entropy_value = -sum((count / total) * np.log2(count / total) for count in label_counts.values())
    
    return entropy_value

# Majority error function
def majority_error(data):
    labels = data[label]
    label_counts = Counter(labels)
    most_common_label_count = label_counts.most_common(1)[0][1]
    total = len(labels)
    
    majorityerror = 1 - (most_common_label_count / total)
    
    return majorityerror

# Gini index function
def gini_index(data):
    labels = data[label]
    label_counts = Counter(labels)
    total = len(labels)
    
    gini_value = 1 - sum((count / total) ** 2 for count in label_counts.values())
    
    return gini_value

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
    # Ask the user to specify the maximum depth
    max_depth = input("Enter the maximum depth of the tree (1 - 6): ").strip()
    
    if max_depth == '':
        max_depth = 6  # Default to be maximum
    else:
        max_depth = int(max_depth)
    
    heuristics = ['entropy', 'majority_error', 'gini']
    
    print(f"Results for max depth: {max_depth}\n")
    # Loop through all three heuristics
    for heuristic in heuristics:
        print(f"Running ID3 algorithm with heuristic: {heuristic}")
        
        # Build a decision tree using the ID3 algorithm
        tree = id3_algorithm(train_data, attributes, label, max_depth=max_depth, heuristic=heuristic)
        
        # Calculate training error
        training_error = calculate_error(tree, train_data)
        testing_error = calculate_error(tree, test_data)
        
        print(f"Training Error: {training_error:.4f}")
        print(f"Testing Error: {testing_error:.4f}\n")


if __name__ == '__main__':
    main()