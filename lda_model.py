#!/usr/bin/env python3

# Tests for logistic_regression.py using a linear discriminative analysis model
import numpy as np
import pandas as pd
from logistic_regression import evaluate, mean_cross_entropy
import matplotlib.pyplot as plt


def visualize_lda(X, y):
    fig,ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y.astype(np.int64).tolist(), s=100, alpha=0.5)
    line_x = np.linspace(-5,5,100)
    line_y = - line_x
    ax.plot(line_x,line_y)
    plt.show()


if __name__ == '__main__':
    # generate fake data in accordance with LDA model
    n = 100  # number of data points in each group
    X1 = np.random.randn(n,2) + np.array([2,2])
    X2 = np.random.randn(n,2) + np.array([-2,-2])
    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(n), np.zeros(n)))
    print(X.shape)
    print(y.shape)

    random_weights = np.random.randn(X.shape[1] + 1)
    print(random_weights.shape)
    print(X.dtype)
    outputs = evaluate(X, random_weights)
    print(outputs.shape)
    print(mean_cross_entropy(outputs, y))
    outputs = evaluate(X, np.array([0, 4, 4]))
    print(mean_cross_entropy(outputs, y))

    visualize_lda(X, y)
