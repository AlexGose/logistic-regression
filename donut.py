#!/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import batch_gradient_ascent


def get_train_data(inner_radius=2, outer_radius=4, N=1000, sigma=1):
    r = np.array([inner_radius] * int(N//2) + [outer_radius] * int(N//2))
    r = r.astype(np.float64)
    r += sigma * np.random.randn(N)
    theta = 2 * np.pi * np.random.random(N)

    X = np.zeros((N,2))
    X[:,0] = r * np.cos(theta)  # cartesian coordinates
    X[:,1] = r * np.sin(theta)

    y = np.array([0] * int(N//2) + [1] * int(N//2))
    return X, y, r


if __name__ == '__main__':
    X, y, r = get_train_data(inner_radius=5, outer_radius=10)

    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

    weights, train_costs, _ = batch_gradient_ascent(X, y, verbose=True, lr=0.001,
                                                 max_iters=1000, step_size=100)
    print(weights)

    X = np.hstack((X, r.reshape((-1,1))))  # add a feature: radius
    weights, train_costs, _ = batch_gradient_ascent(X, y, verbose=True, lr=0.001,
                                                 max_iters=1000, step_size=100)
    print(weights)

