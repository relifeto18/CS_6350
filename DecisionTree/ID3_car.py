#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter

# Load the datasets
train_data = pd.read_csv('./data/car/train.csv', header=None)
test_data = pd.read_csv('./data/car/test.csv', header=None)

# Get attribute descriptions from data-desc.txt
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
label = 'label'

# Add one row in the datasets
train_data.columns = attributes + [label]
test_data.columns = attributes + [label]

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

# Function to split data based on an attribute
def split_data(data, attribute):
    values = data[attribute].unique()
    splits = {value: data[data[attribute] == value] for value in values}
    
    return splits

# Function to choose the best attribute
def choose_best_attribute(data, attributes, heuristic):
    # Base heuristic
    if heuristic == 'entropy':
        base_heuristic_value = entropy(data)
    elif heuristic == 'majority_error':
        base_heuristic_value = majority_error(data)
    elif heuristic == 'gini':
        base_heuristic_value = gini_index(data)
    
    best_gain = 0
    best_attribute = None
    
    for attribute in attributes:
        splits = split_data(data, attribute)
        weighted_average = sum((len(subset) / len(data)) * (entropy(subset) if heuristic == 'entropy'
                                                            else majority_error(subset) if heuristic == 'majority_error'
                                                            else gini_index(subset))
                               for subset in splits.values())
        
        gain = base_heuristic_value - weighted_average
        
        # Update the best attribute
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
        
    return best_attribute

# Build the ID3 tree
class TreeNode:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.children = {}

def build_tree(data, attributes, heuristic, max_depth, current_depth=0):
    labels = data[label]
    
    if len(set(labels)) == 1:
        return TreeNode(label=labels.iloc[0])
    
    if not attributes or current_depth == max_depth:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])
    
    best_attribute = choose_best_attribute(data, attributes, heuristic)
    
    # Check if no valid attribute was found
    if best_attribute is None:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])
    
    tree = TreeNode(attribute=best_attribute)
    
    splits = split_data(data, best_attribute)
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Recursively call
    for value, subset in splits.items():
        if subset.empty:
            tree.children[value] = TreeNode(label=Counter(labels).most_common(1)[0][0])
        else:
            tree.children[value] = build_tree(subset, remaining_attributes, heuristic, max_depth, current_depth + 1)
    
    return tree

# Function to predict the label for a single example
def predict(tree, example):
    if tree.label is not None:
        return tree.label
    
    attribute_value = example[tree.attribute]
    if attribute_value in tree.children:
        return predict(tree.children[attribute_value], example)
    else:
        return None

# Function to evaluate the decision tree on a dataset
def evaluate(tree, data):
    predictions = data.apply(lambda row: predict(tree, row), axis=1)
    accuracy = (predictions == data[label]).mean()
    error = 1 - accuracy
    
    return error

# Train and test the decision tree using different heuristics and depths
def run_experiment(max_depth):
    heuristics = ['entropy', 'majority_error', 'gini']
    results = {}
    
    for heuristic in heuristics:
        tree = build_tree(train_data, attributes, heuristic, max_depth)
        train_error  = evaluate(tree, train_data)
        test_error = evaluate(tree, test_data)
        results[heuristic] = (train_error, test_error)
    
    return results


if __name__ == "__main__":
    # Get user input for max_depth
    max_depth = int(input("Set the maximum tree depth (1-6): ").strip())

    # Run experiment with specified inputs
    results = run_experiment(max_depth)

    # Output the results
    print(f"Results for max depth: {max_depth}")
    for heuristic, (train_error, test_error) in results.items():
        print(f"Heuristic: {heuristic}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")


# def run_experiment(heuristics, depths):
#     results = {}
#     for heuristic in heuristics:
#         results[heuristic] = []
#         for depth in depths:
#             tree = build_tree(train_data, attributes, heuristic, depth)
#             train_error = evaluate(tree, train_data)
#             test_error = evaluate(tree, test_data)
#             results[heuristic].append((depth, train_error, test_error))
    
#     return results

# # Run experiments
# heuristics = ['entropy', 'majority_error', 'gini']
# depths = range(1, 7)
# results = run_experiment(depths)

# # Display results
# for heuristic in results:
#     print(f"Heuristic: {heuristic}")
#     for depth, train_acc, test_acc in results[heuristic]:
#         print(f"Depth: {depth}, Train Error: {train_acc:.4f}, Test Error: {test_acc:.4f}")