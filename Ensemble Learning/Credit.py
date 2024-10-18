#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import AdaBoost, Bagging, RandomForest

label = 'default payment next month'
file_path = "./data/credit_card/test.csv"

# Function to binarize numerical features based on the median
def binarize_numerical_features(data, numerical_attributes):
    for col in numerical_attributes:
        median_value = data[col].median()
        data[col] = np.where(data[col] > median_value, 1, 0)
        
    return data

def main():       
    data_reprocessed = pd.read_csv(file_path, header=1).drop(columns=['ID'])
    attributes = data_reprocessed.columns.tolist()[:-1]
    numerical_attributes = attributes

    # Split into training (24,000) and testing (6,000)
    train_data, test_data = train_test_split(data_reprocessed, train_size=24000, test_size=6000, random_state=42)
 
    # Binarize the numerical columns in both train and test data
    train_data = binarize_numerical_features(train_data, numerical_attributes)
    test_data = binarize_numerical_features(test_data, numerical_attributes)
    
    T = 500
    
    # Run AdaBoost on the training data
    classifiers, alphas, training_errors, testing_errors, stump_training_errors, stump_testing_errors = AdaBoost.adaboost_with_error_tracking(
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


if __name__ == '__main__':
    main()