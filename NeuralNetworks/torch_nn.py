import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_func, initialization):
        super(NeuralNetwork, self).__init__()
        
        layers = []

        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Apply initialization
            if initialization == "xavier":
                nn.init.xavier_uniform_(layers[-1].weight)
            elif initialization == "he":
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")
            
            # Add activation function
            if activation_func == "tanh":
                layers.append(nn.Tanh())
            elif activation_func == "relu":
                layers.append(nn.ReLU())
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Load and preprocess dataset
def load_data():
    train_data = pd.read_csv('./data/bank-note/bank-note/train.csv', header=None)
    test_data = pd.read_csv('./data/bank-note/bank-note/test.csv', header=None)

    # Split features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), \
           torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Train the model
def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=100):
    train_errors = []
    test_errors = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        # Record training error
        train_errors.append(loss.item())

        # Testing
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
            test_errors.append(test_loss.item())

    return train_errors, test_errors

# Main function to test different configurations
def main():
    depths = [3, 5, 9]  # Number of layers
    widths = [5, 10, 25, 50, 100]  # Number of neurons per layer
    activations = ["tanh", "relu"]
    initializations = {"tanh": "xavier", "relu": "he"}
    epochs = 100
    learning_rate = 1e-3

    X_train, y_train, X_test, y_test = load_data()
    results = []

    for activation in activations:
        for depth in depths:
            for width in widths:
                print(f"Training with Activation: {activation}, Depth: {depth}, Width: {width}")
                hidden_sizes = [width] * depth
                
                # Initialize the model
                model = NeuralNetwork(
                    input_size=X_train.shape[1],
                    hidden_sizes=hidden_sizes,
                    activation_func=activation,
                    initialization=initializations[activation]
                )

                # Define loss and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the model
                train_errors, test_errors = train_model(
                    model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs
                )

                # Record final errors
                final_train_error = train_errors[-1]
                final_test_error = test_errors[-1]
                results.append({
                    "Activation": activation,
                    "Depth": depth,
                    "Width": width,
                    "Final Train Error": final_train_error,
                    "Final Test Error": final_test_error,
                })

    # Display results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
