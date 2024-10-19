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

# Function to calculate bias, variance, and squared error
def calculate_bias_variance(predictions, true_labels):
    # Calculate the mean prediction
    mean_prediction = np.mean(predictions, axis=0)
    
    # Calculate bias (how far the average prediction is from the true value)
    bias = np.mean((mean_prediction - true_labels) ** 2)
    
    # Calculate variance (how predictions vary across different models)
    variance = np.mean(np.var(predictions, axis=0))
    
    # Squared error is the sum of bias^2 and variance
    squared_error = bias + variance
    
    return bias, variance, squared_error

def convert_to_numeric(predictions):
    return np.array([1 if pred == 'yes' else 0 for pred in predictions])

# Bagged Trees Algorithm with Bias-Variance Decomposition
def random_forest_bias_variance(train_data, test_data, num_trees, subset_sizes, label, attributes, num_iterations, sample_size):
    results = []

    true_labels = convert_to_numeric(test_data[label].values)
    
    for subset_size in subset_sizes:
        all_bagged_predictions = []
        all_single_tree_predictions = []
        
        for run in range(num_iterations):
            print("subset_size, num_trees:", subset_size, run)
            sampled_train_data = train_data.sample(n=sample_size, replace=False)
            
            # Train the bagged trees
            trees = []
            for _ in range(num_trees):
                bootstrapped_data = bootstrap_sample(sampled_train_data)
                tree = id3_algorithm(bootstrapped_data, attributes, subset_size, label)
                trees.append(tree)
            
            # Store the predictions from bagged trees
            bagged_predictions = np.array([convert_to_numeric([predict(tree, instance) for _, instance in test_data.iterrows()]) for tree in trees])
            all_bagged_predictions.append(np.mean(bagged_predictions, axis=0))
            
            # Store predictions for single tree
            single_tree_predictions = convert_to_numeric([predict(trees[0], instance) for _, instance in test_data.iterrows()])
            all_single_tree_predictions.append(single_tree_predictions)
        
        # Convert to arrays for bias-variance calculations
        all_bagged_predictions = np.array(all_bagged_predictions)
        all_single_tree_predictions = np.array(all_single_tree_predictions)
        
        # Bias and variance for bagged trees
        bias_bagged, variance_bagged, squared_error_bagged = calculate_bias_variance(all_bagged_predictions, true_labels)
        
        # Bias and variance for single trees
        bias_single_tree, variance_single_tree, squared_error_single_tree = calculate_bias_variance(all_single_tree_predictions, true_labels)
        
        # Store the results for this subset size
        results.append((subset_size, bias_single_tree, variance_single_tree, squared_error_single_tree,
                        bias_bagged, variance_bagged, squared_error_bagged))
    
    return results
    
def parallel_random_forest_bias_variance(train_data, test_data, num_trees, subset_sizes, label, attributes, num_iterations, sample_size):
    # Dictionary to store the history of metrics across iterations for each subset size
    history = {
        'bias_single_tree': {subset_size: [] for subset_size in subset_sizes},
        'variance_single_tree': {subset_size: [] for subset_size in subset_sizes},
        'squared_error_single_tree': {subset_size: [] for subset_size in subset_sizes},
        'bias_bagged': {subset_size: [] for subset_size in subset_sizes},
        'variance_bagged': {subset_size: [] for subset_size in subset_sizes},
        'squared_error_bagged': {subset_size: [] for subset_size in subset_sizes}
    }

    true_labels = convert_to_numeric(test_data[label].values)
    
    for subset_size in subset_sizes:
        all_bagged_predictions = []
        all_single_tree_predictions = []
        
        # Parallelize the loop over iterations using joblib and tqdm for progress
        with tqdm(total=num_iterations, desc=f"Subset size {subset_size}") as pbar:
            for run in range(num_iterations):
                sampled_train_data = train_data.sample(n=sample_size, replace=False)
                
                # Train the bagged trees
                trees = Parallel(n_jobs=10)(delayed(id3_algorithm)(bootstrap_sample(sampled_train_data), attributes, subset_size, label)
                                            for _ in range(num_trees))
                
                # Store the predictions from bagged trees
                bagged_predictions = np.array([convert_to_numeric([predict(tree, instance) for _, instance in test_data.iterrows()]) for tree in trees])
                single_tree_predictions = convert_to_numeric([predict(trees[0], instance) for _, instance in test_data.iterrows()])

                # Convert predictions to arrays for bias-variance calculations
                all_bagged_predictions.append(np.mean(bagged_predictions, axis=0))
                all_single_tree_predictions.append(single_tree_predictions)
                
                # Convert to arrays for bias-variance calculations
                all_bagged_predictions_arr = np.array(all_bagged_predictions)
                all_single_tree_predictions_arr = np.array(all_single_tree_predictions)
                
                # Bias and variance for bagged trees
                bias_bagged, variance_bagged, squared_error_bagged = calculate_bias_variance(all_bagged_predictions_arr, true_labels)
                
                # Bias and variance for single trees
                bias_single_tree, variance_single_tree, squared_error_single_tree = calculate_bias_variance(all_single_tree_predictions_arr, true_labels)
                
                # Append to history (storing values for each subset size and iteration)
                history['bias_single_tree'][subset_size].append(bias_single_tree)
                history['variance_single_tree'][subset_size].append(variance_single_tree)
                history['squared_error_single_tree'][subset_size].append(squared_error_single_tree)
                history['bias_bagged'][subset_size].append(bias_bagged)
                history['variance_bagged'][subset_size].append(variance_bagged)
                history['squared_error_bagged'][subset_size].append(squared_error_bagged)
                
                pbar.update(1)  # Update progress bar after each iteration

    return history


def plot_history_across_iterations(history, num_iterations):
    subset_sizes = [2, 4, 6]
    iterations = list(range(1, num_iterations + 1))
    
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Bias
    for subset_size in subset_sizes:
        axes[0].plot(iterations, history['bias_single_tree'][subset_size], label=f"Single Tree Bias (subset {subset_size})", marker='o')
        axes[0].plot(iterations, history['bias_bagged'][subset_size], label=f"Bagged Trees Bias (subset {subset_size})", marker='x')
    
    axes[0].set_title('Bias across Iterations')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Bias')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Variance
    for subset_size in subset_sizes:
        axes[1].plot(iterations, history['variance_single_tree'][subset_size], label=f"Single Tree Variance (subset {subset_size})", marker='o')
        axes[1].plot(iterations, history['variance_bagged'][subset_size], label=f"Bagged Trees Variance (subset {subset_size})", marker='x')
    
    axes[1].set_title('Variance across Iterations')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Variance')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Squared Error
    for subset_size in subset_sizes:
        axes[2].plot(iterations, history['squared_error_single_tree'][subset_size], label=f"Single Tree Squared Error (subset {subset_size})", marker='o')
        axes[2].plot(iterations, history['squared_error_bagged'][subset_size], label=f"Bagged Trees Squared Error (subset {subset_size})", marker='x')
    
    axes[2].set_title('Squared Error across Iterations')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Squared Error')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the figure
    plt.show()
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
    num_iterations = 100
    sample_size = 1000
    feature_subset_sizes = [2, 4, 6]
    
    # results = random_forest_bias_variance(train_data, test_data, num_trees, feature_subset_sizes, label, attributes, num_iterations, sample_size)
    history = parallel_random_forest_bias_variance(train_data, test_data, num_trees, feature_subset_sizes, label, attributes, num_iterations, sample_size)
    
    # Print results for each subset size (after all iterations)
    for subset_size in feature_subset_sizes:
        bias_single_tree = history['bias_single_tree'][subset_size][-1]  # Final bias for single tree at last iteration
        variance_single_tree = history['variance_single_tree'][subset_size][-1]  # Final variance for single tree
        squared_error_single_tree = history['squared_error_single_tree'][subset_size][-1]  # Final squared error for single tree
        
        bias_bagged = history['bias_bagged'][subset_size][-1]  # Final bias for bagged trees at last iteration
        variance_bagged = history['variance_bagged'][subset_size][-1]  # Final variance for bagged trees
        squared_error_bagged = history['squared_error_bagged'][subset_size][-1]  # Final squared error for bagged trees
        
        print(f"Subset Size: {subset_size}")
        print(f"Single Tree Bias: {bias_single_tree:.4f}, Variance: {variance_single_tree:.4f}, Squared Error: {squared_error_single_tree:.4f}")
        print(f"Bagged Trees Bias: {bias_bagged:.4f}, Variance: {variance_bagged:.4f}, Squared Error: {squared_error_bagged:.4f}")
        print("--------------------------------------------------------")
    
    # Plot the results
    plot_history_across_iterations(history, num_iterations)

if __name__ == '__main__':
    main()