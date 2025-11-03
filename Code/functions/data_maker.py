from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def f_true(x):
    """
    Return 1D Runge function
    """
    return 1.0 / (1.0 + 25.0 * x**2)

def make_data(n, seed=seed):
    """
    Makes a data set of length n over the Runge function
    for x in (-1, 1). Includes stochastic noise.

    Creates train and test data sets
    """

    x = np.linspace(-1, 1, n)
    x = x.reshape(-1, 1)

    scaler = StandardScaler(with_std=False)
    scaler.fit(x)
    x_s = scaler.transform(x)

    y_clean = f_true(x_s)
    y_clean = f_true(x_s).flatten()
    y = y_clean + np.random.normal(0, 0.1, n)

    x_train, x_test, y_train, y_test = train_test_split(
        x_s, y, test_size=0.2, random_state=seed, shuffle=True
    )

    train = (x_train, y_train)
    test = (x_test, y_test)
    full = (x_s, y, y_clean)
    return train, test, full