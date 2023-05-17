import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(- z))


def evaluate(inputs, weights):
    """
    Evaluate a logistic regression model at the input data points `inputs`
    using `weights`.  Return the probability that each data point
    should be classified as belonging to class 1, as opposed to class 0.

    inputs -  N x F 2D Numpy array representing the input data
    weights - F + 1 1D Numpy array representing the model weights.
              weights[0] is the bias term.  weights[i] corresponds to
              the ith feature.

    Here, N represents the number of input data points and F is the
    number of input features.
    """
    return sigmoid( add_bias_feature(inputs) @ weights )


def predict(inputs, weights):
    return np.round(evaluate(inputs, weights))


def classification_rate(predictions, targets):
    return (predictions == targets).mean()


def cross_entropy_loss(output, target):
    return - np.log(output) if target == 1 else -np.log(1-output)


def mean_cross_entropy(outputs, targets):
    return np.vectorize(cross_entropy_loss)(outputs, targets).mean()


def ridge_regression_cost(outputs, targets, weights, reg_param=0.01):
    regularization = reg_param * np.linalg.norm(weights, 2)
    return mean_cross_entropy(outputs, targets) + regularization


def add_bias_feature(inputs):
    """
    Prepend a column of ones to `inputs`
    """
    return np.hstack((np.ones((inputs.shape[0],1)), inputs))


def gradient(inputs, targets, weights):
    """
    Returns the gradient of the log likelihood w.r.t. the weights

    inputs - N x F 2d Numpy array representing the input data
    targets - N x 1 1D Numpy array representing the targets
    weights - F + 1 1D Numpy array representing the model weights

    Here, N represents the number of input data points and F is the
    number of input features.
    """
    return ((targets - evaluate(inputs, weights)).reshape((-1,1))
            * add_bias_feature(inputs)).sum(axis=0)


def batch_gradient_ascent(train_inputs, train_targets, initial_weights=None,
                          lr=0.01, verbose=False, max_iters=1000,
                          step_size=20, test_inputs=None, test_targets=None):
    """
    Full batch gradient ascent to maximize likelihood

    Returns the weights at the last iteration, train costs, and test costs
    """
    test_data_exists = False
    if np.any(test_inputs) and np.any(test_targets):
        test_data_exists = True
    if not np.any(initial_weights):
        weights = np.random.randn(train_inputs.shape[1] + 1)
    else:
        weights = initial_weights

    train_costs = []
    test_costs = []
    for it in range(max_iters):

        weights = weights + lr * gradient(train_inputs, train_targets, weights)

        outputs = evaluate(train_inputs, weights)
        train_cost = mean_cross_entropy(outputs, train_targets)
        train_costs.append(train_cost)

        if test_data_exists:
            test_outputs = evaluate(test_inputs, weights)
            test_cost = mean_cross_entropy(test_outputs, test_targets)
            test_costs.append(test_cost)

        if verbose and it % step_size == step_size - 1:
            print(f"{it+1}/{max_iters}: train cost = {train_cost}")
            if test_data_exists:
                print(f"    test cost = {test_cost}")

    if verbose:
        train_predictions = predict(train_inputs, weights)
        final_train_accuracy = classification_rate(train_predictions,
                                                   train_targets)
        print(f"Final train accuracy = {final_train_accuracy}")
        if test_data_exists:
            test_predictions = predict(test_inputs, weights)
            final_test_accuracy = classification_rate(test_predictions,
                                                      test_targets)
            print(f"Final test accuracy = {final_test_accuracy}")

    return weights, train_costs, test_costs


if __name__ == '__main__':
    """
    Test code
    """
    print(sigmoid(0))

    X = np.random.randn(20,2) # generate random input data
    w = np.random.randn(3)

    print(X.shape)
    print(evaluate(X,w))
    print(predict(X,w))

    print(cross_entropy_loss(.9, 1))
    print(cross_entropy_loss(.1, 0))
    print(mean_cross_entropy(np.array([.9, .1]), np.array([1, 0])))
    print(ridge_regression_cost(np.array([.9, .1]), np.array([1, 0]), w))

    y = np.random.choice([0, 1], size=20)  # random labels
    print(gradient(X, y, w))

    print(batch_gradient_ascent(X, y, lr=0.01, verbose=True, max_iters=100))
