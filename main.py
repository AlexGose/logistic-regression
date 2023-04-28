import pandas as pd
import numpy as np
from logistic_regression import predict, classification_rate


def standardize(data_frame, column):
    diff = data_frame[column] - data_frame[column].mean()
    data_frame[column] = diff / data_frame[column].std()
    return data_frame


if __name__ == '__main__':
    df = pd.read_csv("ecommerce_data.csv")
    df = pd.get_dummies(df, columns=['time_of_day'])  ## one-hot encode

    df = df[df['user_action'] <2]  # binary classification

    df = standardize(df, 'n_products_viewed')
    df = standardize(df, 'visit_duration')

    df = df.sample(frac=1, replace=False, axis=0)  # shuffle rows

    print(df)
    print(df.columns)

    # split inputs and outputs
    X = df.drop('user_action',axis=1).to_numpy().astype(np.float32)
    y = df['user_action'].to_numpy().astype(np.float32)

    # train and test split
    X_train = X[:-100]
    y_train = y[:-100]
    X_test = X[-100:]
    y_test = y[-100:]

    print(f"X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"X_test {X_test.shape}, y_test {y_test.shape}")

    weights = np.random.randn(X_train.shape[1] + 1)
    preds = predict(X_train, weights)
    print(preds)
    print(classification_rate(preds, y_train))

