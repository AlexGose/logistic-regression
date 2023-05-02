#!/usr/bin/env python3
import numpy as np
import pandas as pd
from logistic_regression import evaluate

if __name__ == '__main__':
    # generate fake data in accordance with LDA model
    n = 100  # number of data points in each group
    X1 = np.random.randn(n,2) + np.array([2,2])
    X2 = np.random.randn(n,2) + np.array([-2,-2])
    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(n), np.zeros(n)))
    print(X.shape)
    print(y.shape)
