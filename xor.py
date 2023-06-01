import numpy as np
from logistic_regression import batch_gradient_ascent, add_interaction_feature


if __name__ == '__main__':
    """
    Fit the XOR function
    """
    X = np.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    print(f"{X=}")
    X = add_interaction_feature(X, 0, 1)
    print(f"{X=}")
    y = np.array([0, 1, 1, 0])

    weights, train_costs, _ = batch_gradient_ascent(X, y, verbose=True,
                                                    max_iters=1000, 
                                                    step_size=100)
    print(weights)

