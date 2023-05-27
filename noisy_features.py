#!/bin/env python3

import numpy as np
from logistic_regression import batch_gradient_ascent, sigmoid
from logistic_regression import add_bias_feature
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.random.randn(200,20)  # generate train data
    weights = np.array([1, -.5, .5] + [0] * 18)  # true weights
    y = np.round(sigmoid(add_bias_feature(X) @ weights + np.random.randn(200)
                         * 0.5))

    X_test = np.random.randn(50,20)  # generate test data
    y_test = np.round(sigmoid(add_bias_feature(X_test) @ weights 
                         + np.random.randn(50) * 0.5))

    weights1, train_cost1, test_cost1 = batch_gradient_ascent(X, y, verbose=True, 
                                   test_inputs=X_test, test_targets=y_test,
                                   max_iters=100, l1=True, l1_param=7)
    weights2, train_cost2, test_cost2 = batch_gradient_ascent(X, y, verbose=True, 
                                   max_iters=100, test_inputs=X_test, 
                                   test_targets=y_test)

    plt.plot(weights1, label='Lasso')
    plt.plot(weights2, label='Without Lasso')
    plt.legend()
    plt.title('Weights')
    plt.xlabel('Weight Index')
    plt.show()

    plt.plot(train_cost1, label='Lasso')
    plt.plot(train_cost2, label='Without Lasso')
    plt.legend()
    plt.title('Training Cost')
    plt.xlabel('Iteration')
    plt.show()

    plt.plot(test_cost1, label='Lasso')
    plt.plot(test_cost2, label='Without Lasso')
    plt.legend()
    plt.title('Test Cost')
    plt.xlabel('Iteration')
    plt.show()
