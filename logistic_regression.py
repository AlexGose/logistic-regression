import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(- z))


def evaluate(inputs, weights):
    """
    Evaluate a logistic regression model at the input data points `inputs`
    using `weights`.  Return the probability that each data point
    should be classified as belonging to class 1, as opposed to class 0.

    inputs -  N x F 2D Numpy array representing the input data
    weights - N + 1 1D Numpy array representing the model outputs.
              weights[0] is the bias term.  weights[i] corresponds to
              the ith feature.
    
    Here, N represents the number of input data points and F is the
    number of input features.  
    """
    N = inputs.shape[0]
    return sigmoid( np.hstack((np.ones((N,1)),inputs)) @ weights )


if __name__ == '__main__':
    """
    Test code
    """
    print(sigmoid(0))
    
    # generate random input data (20 data points with 2 features)
    X = np.random.randn(20,2)
    w = np.random.randn(3)
    
    print(X.shape)
    print(evaluate(X,w))
