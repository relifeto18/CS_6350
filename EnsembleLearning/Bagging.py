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
    
    # Calculate information gain or reduction in Gini index or majority error for each attribute
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
    # Check if all labels are the same (pure node)
    if len(Counter(data[label])) == 1:
        return {'label': data[label].iloc[0]}
    
    # If there are no attributes left to split, return the majority label
    if not attributes:
        majority_label = Counter(data[label]).most_common(1)[0][0]
        return {'label': majority_label}
    
    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(data, attributes)
    
    # Create a node for the best attribute
    tree = {best_attribute: {}}
    
    # Split the data based on the best attribute
    splits = split_data(data, best_attribute)
    
    # Remove the best attribute from the available attributes
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    # Recursively build the tree for each subset
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
        # Return the most common label if the value is not found
        find_most_common_label(tree)
    
def find_most_common_label(subtree):
    if 'label' in subtree:
        return subtree['label']
    
    # Recursively collect labels from all branches
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

#########################################################
#########################################################
def bootstrap_sample(data):
    sample = data.sample(frac=1, replace=True)
    return sample

def bagged_trees(data, attributes, label, num_trees):
    trees = []
    for _ in range(num_trees):
        bootstrap_data = bootstrap_sample(data)
        tree = id3_algorithm(bootstrap_data, attributes, label)
        trees.append(tree)
    return trees

def predict_with_bagging(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    majority_label = Counter(predictions).most_common(1)[0][0]
    return majority_label

def calculate_bagging_error(trees, data):
    incorrect_predictions = 0
    for _, instance in data.iterrows():
        if predict_with_bagging(trees, instance) != instance[label]:
            incorrect_predictions += 1
    return incorrect_predictions / len(data)

def sample_without_replacement(data, n_samples):
    return data.sample(n=n_samples, replace=False)

label_mapping = {
    'yes': 1,
    'no': 0
}

# Make sure the label is converted to numerical
def predict_with_numerical_labels(tree, instance, label_mapping):
    predicted_label = predict(tree, instance)
    
    if predicted_label is None:
        return label_mapping['no']
    return label_mapping[predicted_label]

def sample_bagged_trees(data, attributes, label, num_trees, n_samples):
    trees = []
    for _ in range(num_trees):
        sampled_data = sample_without_replacement(data, n_samples)
        tree = id3_algorithm(sampled_data, attributes, label)
        trees.append(tree)
    return trees

def calculate_bias_variance(trees, data, label, label_mapping, single_tree=False):
    n_trees = len(trees)
    bias = 0
    variance = 0
    gse = 0

    if single_tree:
        # For single tree, predictions are from one tree only across different test examples
        predictions = np.array([predict_with_numerical_labels(trees[0], instance, label_mapping) for _, instance in data.iterrows()])
        
        # Compute true labels for the test examples
        true_labels = np.array([label_mapping[instance[label]] for _, instance in data.iterrows()])
        
        # Bias calculation
        avg_prediction = np.mean(predictions)
        bias = np.mean((avg_prediction - true_labels) ** 2)
        
        # Variance calculation (calculate variance of predictions across test examples)
        variance = np.var(predictions)

    else:
        # For bagged trees, calculate across all trees
        for _, instance in data.iterrows():
            # Get predictions from all trees and convert to numerical labels
            predictions = np.array([predict_with_numerical_labels(tree, instance, label_mapping) for tree in trees])
            true_label = label_mapping[instance[label]]

            # Bias calculation
            avg_prediction = np.mean(predictions)
            bias += (avg_prediction - true_label) ** 2

            # Variance calculation (across different trees' predictions for the same instance)
            variance += np.var(predictions)

        # Average the bias and variance over all test examples
        bias /= len(data)
        variance /= len(data)

    # General squared error
    gse = bias + variance

    return bias, variance, gse

def run_experiment(train_data, test_data, attributes, label, num_trees=500, n_samples=1000, n_repeats=100):
    # Store results
    single_tree_bias_list = []
    single_tree_variance_list = []
    single_tree_gse_list = []

    bagged_tree_bias_list = []
    bagged_tree_variance_list = []
    bagged_tree_gse_list = []

    for i in range(n_repeats):
        print(i)
        # Sample without replacement for bagging
        trees = sample_bagged_trees(train_data, attributes, label, num_trees, n_samples)

        # Pick the first tree from each bagged run (single decision trees)
        single_trees = [trees[0] for _ in range(n_repeats)]

        # Calculate bias, variance, and GSE for the single tree learner
        bias, variance, gse = calculate_bias_variance(single_trees, test_data, label, label_mapping, single_tree=True)
        single_tree_bias_list.append(bias)
        single_tree_variance_list.append(variance)
        single_tree_gse_list.append(gse)

        # Calculate bias, variance, and GSE for the bagged trees
        bias, variance, gse = calculate_bias_variance(trees, test_data, label, label_mapping)
        bagged_tree_bias_list.append(bias)
        bagged_tree_variance_list.append(variance)
        bagged_tree_gse_list.append(gse)

    # Return the final average bias, variance, and GSE for both single tree and bagged trees
    return {
        'single_tree': {
            'bias': single_tree_bias_list,
            'variance': single_tree_variance_list,
            'gse': single_tree_gse_list
        },
        'bagged_trees': {
            'bias': bagged_tree_bias_list,
            'variance': bagged_tree_variance_list,
            'gse': bagged_tree_gse_list
        }
    }

# Plotting function
def plot_bias_variance_results(results):
    # Get the bias, variance, and GSE for both single trees and bagged trees
    single_tree_bias = results['single_tree']['bias']
    single_tree_variance = results['single_tree']['variance']
    single_tree_gse = results['single_tree']['gse']

    bagged_trees_bias = results['bagged_trees']['bias']
    bagged_trees_variance = results['bagged_trees']['variance']
    bagged_trees_gse = results['bagged_trees']['gse']

    # Generate the x-axis (iterations)
    iterations = range(1, len(single_tree_bias) + 1)

    # Plot Bias, Variance, and GSE for Single Tree
    plt.plot(iterations, single_tree_bias, label='Single Tree Bias', marker='o', color='orange')
    plt.plot(iterations, single_tree_variance, label='Single Tree Variance', marker='x', color='brown')
    plt.plot(iterations, single_tree_gse, label='Single Tree GSE', linestyle='--', color='red')

    # Plot Bias, Variance, and GSE for Bagged Trees
    plt.plot(iterations, bagged_trees_bias, label='Bagged Trees Bias', marker='o', color='purple')
    plt.plot(iterations, bagged_trees_variance, label='Bagged Trees Variance', marker='x', color='blue')
    plt.plot(iterations, bagged_trees_gse, label='Bagged Trees GSE', linestyle='--', color='green')

    # Labels and legend
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Bias, Variance, and GSE for Single Tree and Bagged Trees')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    
# Helper function to execute one iteration of the experiment
def experiment_iteration(train_data, test_data, attributes, label, label_mapping, num_trees, n_samples, iteration):
    # Sample without replacement for bagging
    trees = sample_bagged_trees(train_data, attributes, label, num_trees, n_samples)

    # Pick the first tree from the bagged trees (single decision tree)
    single_trees = [trees[0]]

    # Calculate bias, variance, and GSE for the single tree learner
    single_tree_bias, single_tree_variance, single_tree_gse = calculate_bias_variance(single_trees, test_data, label, label_mapping, single_tree=True)

    # Calculate bias, variance, and GSE for the bagged trees
    bagged_tree_bias, bagged_tree_variance, bagged_tree_gse = calculate_bias_variance(trees, test_data, label, label_mapping)
    
    print(f"Completed iteration {iteration}")

    return (single_tree_bias, single_tree_variance, single_tree_gse, 
            bagged_tree_bias, bagged_tree_variance, bagged_tree_gse)
    
# Updated run_experiment to use parallelism
def parallel_run_experiment(train_data, test_data, attributes, label, num_trees=500, n_samples=1000, n_repeats=100, n_jobs=12):
    # Run iterations in parallel using joblib.Parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(experiment_iteration)(train_data, test_data, attributes, label, label_mapping, num_trees, n_samples, i)
        for i in tqdm(range(n_repeats), desc="Running iterations")
    )
    
    # Initialize lists to store results
    single_tree_bias_list = []
    single_tree_variance_list = []
    single_tree_gse_list = []

    bagged_tree_bias_list = []
    bagged_tree_variance_list = []
    bagged_tree_gse_list = []

    # Aggregate the results from all iterations
    for res in results:
        single_tree_bias, single_tree_variance, single_tree_gse, bagged_tree_bias, bagged_tree_variance, bagged_tree_gse = res
        single_tree_bias_list.append(single_tree_bias)
        single_tree_variance_list.append(single_tree_variance)
        single_tree_gse_list.append(single_tree_gse)
        bagged_tree_bias_list.append(bagged_tree_bias)
        bagged_tree_variance_list.append(bagged_tree_variance)
        bagged_tree_gse_list.append(bagged_tree_gse)

    # Return the lists of bias, variance, and GSE for each iteration
    return {
        'single_tree': {
            'bias': single_tree_bias_list,
            'variance': single_tree_variance_list,
            'gse': single_tree_gse_list
        },
        'bagged_trees': {
            'bias': bagged_tree_bias_list,
            'variance': bagged_tree_variance_list,
            'gse': bagged_tree_gse_list
        }
    }
#########################################################
#########################################################

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
    
    # Vary the number of trees
    num_trees_range = list(range(1, 500))
    train_errors = []
    test_errors = []

    for num_trees in num_trees_range:
        print("num_trees: ", num_trees)
        trees = bagged_trees(train_data, attributes, label, num_trees)
        train_error = calculate_bagging_error(trees, train_data)
        test_error = calculate_bagging_error(trees, test_data)
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