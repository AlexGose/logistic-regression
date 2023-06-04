import numpy as np


def sigmoid(z):
    return np.vectorize(_sigmoid)(z)


def _sigmoid(z):
    """
    Numerically stable helper function to avoid overflow
    """
    if z >= 0:
        return 1 / (1 + np.exp(- z))
    else:
        return np.exp(z) / (np.exp(z) + 1)


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


def mean_cross_entropy(outputs, targets, weights=None, reg=False, reg_norm=1,
                       reg_param=0.01):
    if reg:  # with regularization
        regularization = reg_param * np.linalg.norm(weights, reg_norm)
    else:
        regularization = 0
    return np.vectorize(cross_entropy_loss)(outputs, targets).mean() \
           + regularization


def add_bias_feature(inputs):
    """
    Prepend a column of ones to `inputs`
    """
    return np.hstack((np.ones((inputs.shape[0],1)), inputs))


def add_interaction_feature(inputs, col1, col2):
    """
    Append a feature column to `inputs` representing the product of
    each value in column `col1` with the value in column `col2`
    """
    product = inputs[:,col1] * inputs[:,col2]
    return np.hstack((inputs, product.reshape((-1,1))))


def gradient(inputs, targets, weights, l2=False, l2_param=0.01,
             l1=False, l1_param=0.01):
    """
    Returns the gradient of the log likelihood w.r.t. the weights

    inputs - N x F 2D Numpy array representing the input data
    targets - 1D Numpy array of length N representing the targets
    weights - 1D Numpy array of length F + 1 representing the model weights,
              where weights[0] is the bias term
    l2 - (default False) will include L2 regularization if True
    l2_param - (default 0.01) the regularization parameter for L2
               regularization
    l1 - (default False) will include L1 regularization if True
    l1_param - (default 0.01) the regularization parameter for L1
                regularization 

    Here, N represents the number of input data points and F is the
    number of input features.
    """
    output = ((targets - evaluate(inputs, weights)).reshape((-1,1))
            * add_bias_feature(inputs)).sum(axis=0)
    if l2:
        output -= l2_param * weights
    if l1:
        output -= l1_param * np.sign(weights)
    return output


def batch_gradient_ascent(train_inputs, train_targets, initial_weights=None,
                          lr=0.01, verbose=False, max_iters=1000,
                          step_size=20, test_inputs=None, test_targets=None,
                          l2=False, l2_param=0.01, l1=False, l1_param=0.01):
    """
    Full batch gradient ascent to maximize log likelihood

    Returns the weights at the last iteration, train costs, and test costs
    """
    test_data_exists = True
    if test_inputs is None or test_targets is None:
        test_data_exists = False

    if initial_weights is None:
        num_weights = train_inputs.shape[1] + 1
        weights = np.random.randn(num_weights) / np.sqrt(num_weights)
    else:
        weights = initial_weights.astype(np.float64)

    train_costs = []
    test_costs = []
    for it in range(max_iters):

        weights += lr * gradient(train_inputs, train_targets, weights, 
                          l2=l2, l2_param=l2_param, l1=l1, l1_param=l1_param)

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

    np.random.seed(1)
    X = np.random.randn(20,2) # generate random input data
    w = np.random.randn(3)

    print(X.shape)
    print(evaluate(X,w))
    print(predict(X,w))

    print(cross_entropy_loss(.9, 1))
    print(cross_entropy_loss(.1, 0))
    print(mean_cross_entropy(np.array([.9, .1]), np.array([1, 0])))
    print(mean_cross_entropy(np.array([.9, .1]), np.array([1, 0]), w, reg=True,
                             reg_norm=2))
    print(mean_cross_entropy(np.array([.9, .1]), np.array([1, 0]), w, reg=True,
                             reg_norm=1))

    y = np.random.choice([0, 1], size=20)  # random labels
    print(gradient(X, y, w))
    print(gradient(X, y, w, l2=True))

    print(batch_gradient_ascent(X, y, lr=0.01, verbose=True, max_iters=100))
    print(batch_gradient_ascent(X, y, verbose=True, max_iters=100,
          l2=True, l2_param=1))
    print(batch_gradient_ascent(X, y, verbose=True, max_iters=100,
          l1=True, l1_param=1))
    print(batch_gradient_ascent(X, y, initial_weights=np.zeros(3), max_iters=10))
