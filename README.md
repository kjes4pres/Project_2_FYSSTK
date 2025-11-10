# Project 2 FYS-STK4155
### Fall 2025
**Authors:** *Ingvild Olden Bjerkelund, Kjersti Stangeland, Jenny Guldvog, & Sverre Manu Johansen*

This is a collaboration repository for project 2 in FYS-STK4155. The aim of this project is to study both classification and regression problems by developing our own feed-forward neural network (FFNN) code.

### How to install required packages
A `requirements.txt` file is located in the repository. To reproduce our results, use the packages listed here. To install the packages, download the `requirements.txt` file, open your terminal and locate your project repository where you placed the downloaded file, in the command line write "´pip install -r requirements.txt´" or if you're using a conda environment type `conda install --file requirements.txt`.

### Overview of contents
The repository is organized as follows:

Functions and modules used for obtaining the results:
* `Code/functions/activation_funcs.py`: Python module containing functions for different activation functions used in the hidden layers of the neural network, and their derivatives. 
* `Code/functions/cost_functions.py`: Python module containing functions for different cost functions used in theneural network, and their derivatives. 
* `Code/functions/ffnn.py`: Python module containing our neural network class.
* `Code/functions/runge.py`: Module returning the Runge function. 
* `Code/functions/ffnn_lib_funcs.py`: Functions for using different libraries to compare to our FFNN. 

Notebooks for running the code and plotting:
* `Code/main/b.ipynb`: Analysis of our FNNN on the Runge function and comparison to OLS.
* `Code/main/c.ipynb`: Comparison of our FFNN to Sckit-Learn, PyTorch and TensorFlow-Keras. Also comparing our derivatives to Autograd.
* `Code/main/d_and_e.ipynb`: Jupyter notebook which assesses the impact of activation functions, number of hidden layers and number of nodes in the neural network performance on predicting the 1D Runge function. It also investigates the impact of added norms to the perfomance, and compares this to results from Project 1. 
* `Code/main/f.ipynb`: Notebook for analysis of our NN on the MNIST dataset.