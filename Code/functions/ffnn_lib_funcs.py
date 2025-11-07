# Import packages : 
import time
import numpy as np
from typing import Tuple, List
# Our own FFNN : 
from functions.ffnn import *
from functions.cost_functions import *
# Scikit-Learn :
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Tensorflow-Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam     # NB! Remember this definition !!
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2      # NB! Remember this definition !!
# PyTorch : 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# For reproducibility
# Same seed as project 1
np.random.seed(2018)
seed = np.random.seed(2018)
torch.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
tf.random.set_seed(2018)

def FFNN(network_input_size, FFNN_layers, FFNN_act_fun, FFNN_act_fun_der, FFNN_cost_fun, FFNN_cost_fun_der,
         inputs, targets, epochs, learning_rate, batch_size, optimizer, shuffle, beta1, beta2, replace, x_test, y_test):
    # Initialize the network, so that the weights are not remembered
    Regression_FFNN = NeuralNetwork(network_input_size,FFNN_layers,FFNN_act_fun,FFNN_act_fun_der,FFNN_cost_fun,FFNN_cost_fun_der)
    # Start the timer
    start_time = time.time()
    # Train the model
    #Regression_FFNN.train_SGD(inputs, targets, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, optimizer = FFN_solver)
    Regression_FFNN.train_SGD_v2(inputs, targets, epochs, learning_rate, batch_size, optimizer, shuffle, beta1, beta2, replace)
    # Make predictions
    y_pred = Regression_FFNN._feed_forward(x_test)
    # Calculate mean squared error
    mse_proj2  = mse(y_pred, y_test)
    mse_proj22 = mean_squared_error(y_test, y_pred)
    # Calculate elapsed time
    elapsed_time_FFNN = time.time() - start_time
    # Print results
    print(f"Project-2 MSE: {mse_proj2}")
    print(f"Project-2 MSE: {mse_proj22}")
    print(f"Elapsed time: {elapsed_time_FFNN:.2f} seconds")
    return Regression_FFNN, elapsed_time_FFNN, mse_proj2, y_pred

def scikitFFNN(scikit_layer, scikit_act_fun, scikit_solver, scikit_alpha, batch_size, scikit_lr_type, learning_rate,
               epochs, shuffle, seed, tolerance, scikit_verbose, scikit_warm_start, scikit_momentum, scikit_nesterovs_momentum,
               early_stopping, scikit_validation_fraction, beta_1, beta_2, scikit_epsilon, scikit_n_iter_no_change,
               inputs, targets, x_test, y_test):
    # Start the timer
    start_time = time.time()
    # Create and train the Scikit-Learn model --> Multi-layer Perceptron regressor : 
    sklearn_model = MLPRegressor(#loss='squared_error', 
                                hidden_layer_sizes  = scikit_layer, 
                                activation          = scikit_act_fun, # Only on the hidden layers, last have identity as default
                                solver              = scikit_solver,
                                alpha               = scikit_alpha,
                                batch_size          = batch_size,
                                learning_rate       = scikit_lr_type,
                                learning_rate_init  = learning_rate, 
                                # power_t=0.5, # The exponent for inverse scaling learning rate. 
                                # It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.
                                max_iter            = epochs, 
                                shuffle             = shuffle, 
                                random_state        = seed, 
                                tol                 = tolerance, 
                                verbose             = scikit_verbose,
                                warm_start          = scikit_warm_start,
                                momentum            = scikit_momentum,
                                nesterovs_momentum  = scikit_nesterovs_momentum,
                                early_stopping      = early_stopping,
                                validation_fraction = scikit_validation_fraction,
                                beta_1              = beta_1,
                                beta_2              = beta_2,
                                epsilon             = scikit_epsilon,
                                n_iter_no_change    = scikit_n_iter_no_change,
                                # max_fun=15000, # Only used when solver=’lbfgs’. 
                                )
    sklearn_model.fit(inputs, targets.ravel())
    # Predictions
    y_pred_sklearn = sklearn_model.predict(x_test)
    # Calculate mean squared error
    mse_sklearn  = mse(y_pred_sklearn, y_test.flatten())
    mse_sklearn2 = mean_squared_error(y_test, y_pred_sklearn)
    # Calculate elapsed time
    elapsed_time_scikit = time.time() - start_time
    print(f"Scikit-Learn MSE: {mse_sklearn}")
    print(f"Scikit-Learn MSE: {mse_sklearn2}")
    print(f"Elapsed time: {elapsed_time_scikit:.2f} seconds")
    return sklearn_model, elapsed_time_scikit, mse_sklearn, y_pred_sklearn

def kerasFFNN(keras_layers, keras_act_fun, network_input_size, keras_cost_fun, keras_solver, 
              inputs, targets, epochs, batch_size, keras_verbose, shuffle, x_test, y_test):
    # Start the timer
    start_time = time.time()
    # Create a simple feedforward neural network using Keras
    model = Sequential()
    # Add layers to the model
    for i in range(len(keras_layers)):
        if i == 0: 
            # Input layer
            model.add(Dense(keras_layers[i], activation=keras_act_fun[i], input_dim=network_input_size)) 
        else:
            model.add(Dense(keras_layers[i], activation=keras_act_fun[i]))  
    # Compile the model with optimkeras_solverizer
    model.compile(loss=keras_cost_fun, optimizer=keras_solver)
    # Define the prediction function
    @tf.function(reduce_retracing=True)
    def predict_fn(x):
        return model(x)
    # Train the model and capture the loss history
    history = model.fit(inputs, targets, 
                        epochs     = epochs, 
                        batch_size = batch_size, 
                        verbose    = keras_verbose,
                        #validation_split = 0,
                        shuffle     = shuffle
                        )
    # Predictions
    #y_pred_keras = model.predict(x_test).numpy() # OLD
    y_pred_keras = predict_fn(x_test).numpy()  # Convert from tensor to numpy
    # Calculate mean squared error
    mse_keras  = mse(y_pred_keras, y_test)
    mse_keras2 = mean_squared_error(y_test, y_pred_keras)
    # Calculate elapsed time
    elapsed_time_keras = time.time() - start_time
    print(f"Tensorflow-Keras MSE: {mse_keras}")
    print(f"Tensorflow-Keras MSE: {mse_keras2}")
    print(f"Elapsed time: {elapsed_time_keras:.2f} seconds")
    return history, elapsed_time_keras, mse_keras, y_pred_keras

def pytorchFFNN(pytorch_layers, inputs, targets, batch_size, shuffle, learning_rate, epochs,
                x_test, y_test):
    # Start the timer
    start_time = time.time()
    # Device configuration: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define a simple feedforward neural network
    class FeedForwardNN(nn.Module):
        def __init__(self, layers):
            super(FeedForwardNN, self).__init__()
            self.layers = nn.ModuleList()
            # Create hidden layers
            for i in range(len(layers) - 1):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        def forward(self, x):
            # Forward pass through the hidden layers with sigmoid activation
            for layer in self.layers[:-1]:  # All but the last layer
                x = torch.sigmoid(layer(x))
            x = self.layers[-1](x)  # Output layer (linear activation)
            return x
    # Prepare data
    x_tensor = torch.tensor(inputs, dtype=torch.float32).view(-1, 1).to(device)   # Reshape for PyTorch
    y_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device)                  # Reshape and send to device
    # Create a TensorDataset and DataLoader
    dataset     = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # shuffle=True to shuffle the data
    # Initialize the model, loss function and optimizer
    model = FeedForwardNN(pytorch_layers).to(device)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Changed from SGD to Adam
    # List to store loss values
    loss_history = []
    # Training loop
    num_batches = len(data_loader)
    #print(num_batches)
    for epoch in range(epochs):
        model.train()                        # Set model to training mode
        # # Shuffle data if needed (equivalent to your SGD method)
        # if shuffle:
        #     indices = np.random.permutation(inputs.shape[0])
        #     x_tensor = x_tensor[indices]
        #     y_tensor = y_tensor[indices]
        #print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0 # Initialize loss for the epoch
        # Loop over each batch
        for batch_idx, (inputs_batch, targets_batch) in enumerate(data_loader):
            # Logging epoch and batch index
            #print(f"Epoch {epoch + 1}/{epochs}, Iteration {batch_idx + 1}/{len(data_loader)}")
            # Get the data for this batch
            #inputs_batch, targets_batch = next(iter(data_loader))
            inputs_batch = inputs_batch.to(device)  # Move to device
            targets_batch = targets_batch.to(device)  # Move to device
            # Forward pass
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs_batch)  # Forward pass
            loss = criterion(outputs, targets_batch)  # Calculate loss
            loss.backward()  # Backward pass (calculate gradients)
            optimizer.step()  # Update weights
            # Accumulate loss for the epoch
            epoch_loss += loss.item()
            # Store loss for tracking
            #loss_history.append(loss.item())
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        # Store average loss for tracking
        loss_history.append(avg_epoch_loss)
        # # Print:
        # #print(f"Epoch {epoch + 1}/{epochs}, Iteration {batch_idx + 1}/{len(data_loader)}")
        # for batch_idx, (inputs_batch, targets_batch) in enumerate(data_loader):
        # #for inputs_batch, targets_batch in data_loader:
        #     print(f"Epoch {epoch + 1}/{epochs}, Iteration {batch_idx + 1}/{len(data_loader)}")
        #     optimizer.zero_grad()                               # Clear the gradients
        #     outputs = model(inputs_batch.to(device))            # Forward pass
        #     loss = criterion(outputs, targets_batch.to(device)) # Calculate loss
        #     loss.backward()                                     # Backward pass (gradient calculation)
        #     optimizer.step()                                    # Update weights
        #     # Store loss for plotting
        #     loss_history.append(loss.item())
    # Predictions
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)   # Reshape for PyTorch
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # disable gradient calculation for evaluation 
        y_pred_pytorch = model(x_test_tensor).cpu().numpy()  # Move tensor to CPU and convert to numpy
    # Calculate mean squared error
    mse_pytorch = mse(y_pred_pytorch, y_test)  # Calculate MSE with the custom function
    #mse_pytorch  = mse(y_test, y_pred_pytorch)
    mse_pytorch2 = mean_squared_error(y_test, y_pred_pytorch)
    # Calculate elapsed time
    elapsed_time_pytorch = time.time() - start_time
    # Print results
    print(f"PyTorch MSE: {mse_pytorch}")
    print(f"PyTorch MSE: {mse_pytorch2}")
    print(f"Elapsed time: {elapsed_time_pytorch:.2f} seconds")
    return model, elapsed_time_pytorch, loss_history, mse_pytorch, y_pred_pytorch