import sys, os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from functions import *

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# Extract data (features) and target (labels)
X = mnist.data
y = mnist.target
y = np.array([int(i) for i in y])

X = X / 255.0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Reg_nn = NeuralNetwork(X_train.shape[1], [32,1], [sigmoid,softmax], [derivate(sigmoid), derivate(softmax)], cross_entropy, cross_entropy_der)

Reg_nn.train(X_train, y_train, epochs=100)