#!/usr/bin/env python3

# Tests for logistic_regression.py using a linear discriminative analysis model
import numpy as np
import pandas as pd
from logistic_regression import evaluate, mean_cross_entropy, \
                                batch_gradient_ascent
import matplotlib.pyplot as plt


def visualize_lda(X, y):
    fig,ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y.astype(np.int64).tolist(), s=100, alpha=0.5)
    line_x = np.linspace(-5,5,100)
    line_y = - line_x
    ax.plot(line_x,line_y)
    plt.show()


def lda_model_data():
    # Generate fake data in accordance with LDA model
    n = 100  # number of data points in each group
    X1 = np.random.randn(n,2) + np.array([2,2])
    X2 = np.random.randn(n,2) + np.array([-2,-2])
    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(n), np.zeros(n)))
    return X, y


if __name__ == '__main__':
    X, y = lda_model_data()  # avoid train/test split (just testing code)
    print(X.shape)
    print(y.shape)

    random_weights = np.random.randn(X.shape[1] + 1)
    print(random_weights.shape)
    print(X.dtype)
    print(mean_cross_entropy(X, y, random_weights))
    print(mean_cross_entropy(X, y, np.array([0, 4, 4])))

    visualize_lda(X, y)
    batch_gradient_ascent(X, y, random_weights, lr=0.01, verbose=True,
                          max_iters=100)

