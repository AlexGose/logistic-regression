import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("ecommerce_data.csv")
    df = pd.get_dummies(df, columns=['time_of_day'])  ## one-hot encode
    print(df)
    print(df.columns)
