import numpy as np


def sigmoid(z):
    ans = np.zeros(z.shape)
    ans[z<0] = np.exp(z[z<0]) / (np.exp(z[z<0]) + 1)
    ans[z>=0] = 1 / (1 + np.exp(- z[z>=0]))
    return ans


def evaluate(inputs, weights, has_bias_column=False):
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
    if has_bias_column:
        return sigmoid( inputs @ weights )
    else:
        return sigmoid( add_bias_feature(inputs) @ weights )


def predict(inputs, weights, has_bias_column=False):
    return np.round(evaluate(inputs, weights, has_bias_column=has_bias_column))


def classification_rate(predictions, targets):
    return (predictions == targets).mean()


def cross_entropy_loss(feature_weight_products, targets):
    """
    Numerically stable loss calculation
    """
    lses = np.zeros(len(feature_weight_products))
    lses[feature_weight_products <= 0] = np.log( 1 + np.exp(
                    - feature_weight_products[feature_weight_products <= 0]))
    lses[feature_weight_products > 0] = \
            - feature_weight_products[feature_weight_products > 0] + np.log(1
             + np.exp(feature_weight_products[feature_weight_products > 0]))
    ans = np.zeros(len(feature_weight_products))
    ans[targets==1] = lses[targets==1]
    ans[targets==0] = feature_weight_products[targets==0] + lses[targets==0]
    return ans


def mean_cross_entropy(inputs, targets, weights, reg=False, reg_norm=1,
                       reg_param=0.01, has_bias_column=False):
    if reg:  # with regularization
        regularization = reg_param * np.linalg.norm(weights, reg_norm)
    else:
        regularization = 0
    if has_bias_column:
        feature_weight_products = np.dot(inputs , weights)
    else:
        feature_weight_products = np.dot(add_bias_feature(inputs) , weights)
    return cross_entropy_loss(feature_weight_products, targets).mean() \
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
             l1=False, l1_param=0.01, has_bias_column=False):
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
    has_bias_column - (default False) equals true if `inputs` already has a
                      column of ones for the bias term, False otherwise

    Here, N represents the number of input data points and F is the
    number of input features.
    """
    if not has_bias_column:
        output = ((targets - evaluate(inputs, weights)).reshape((-1,1))
                   * add_bias_feature(inputs)).sum(axis=0)
    else:
        output = ((targets - evaluate(inputs, weights, has_bias_column=True))
                   * inputs.T).T.sum(axis=0)
    if l2:
        output -= l2_param * weights
    if l1:
        output -= l1_param * np.sign(weights)
    return output


def batch_gradient_ascent(train_inputs, train_targets, initial_weights=None,
                          lr=0.01, verbose=False, max_iters=1000,
                          step_size=20, test_inputs=None, test_targets=None,
                          l2=False, l2_param=0.01, l1=False, l1_param=0.01,
                          has_bias_column=False):
    """
    Full batch gradient ascent to maximize log likelihood

    Returns the weights at the last iteration, train costs, and test costs
    """
    if not has_bias_column:
        train_inputs = add_bias_feature(train_inputs)

    test_data_exists = True
    if test_inputs is None or test_targets is None:
        test_data_exists = False
    else:
        if not has_bias_column:
            test_inputs = add_bias_feature(test_inputs)

    if initial_weights is None:
        num_weights = train_inputs.shape[1]
        weights = np.random.randn(num_weights) / np.sqrt(num_weights)
    else:
        weights = initial_weights.astype(np.float64)

    train_costs = []
    test_costs = []
    for it in range(max_iters):
        weights += lr * gradient(train_inputs, train_targets, weights, 
                          l2=l2, l2_param=l2_param, l1=l1, l1_param=l1_param,
                          has_bias_column=True)

        train_cost = mean_cross_entropy(train_inputs, train_targets, weights,
                                    has_bias_column=True)
        train_costs.append(train_cost)

        if test_data_exists:
            test_cost = mean_cross_entropy(test_inputs, test_targets, weights,
                                           has_bias_column=True)
            test_costs.append(test_cost)

        if verbose and it % step_size == step_size - 1:
            print(f"{it+1}/{max_iters}: train cost = {train_cost}")
            if test_data_exists:
                print(f"    test cost = {test_cost}")

    if verbose:
        train_predictions = predict(train_inputs, weights,
                                    has_bias_column=True)
        final_train_accuracy = classification_rate(train_predictions,
                                                   train_targets)
        print(f"Final train accuracy = {final_train_accuracy}")
        if test_data_exists:
            test_predictions = predict(test_inputs, weights,
                                       has_bias_column=True)
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
    print(mean_cross_entropy(np.array([[.9, .1]]), np.array([1, 0]), w))
    print(mean_cross_entropy(np.array([[.9, .1]]), np.array([1, 0]), w, reg=True,
                             reg_norm=2))
    print(mean_cross_entropy(np.array([[.9, .1]]), np.array([1, 0]), w, reg=True,
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
