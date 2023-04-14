import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv("ecommerce_data.csv")
    time_of_day = df['time_of_day'].apply(lambda x: 'time_of_day_' + str(x))
    one_hot = pd.get_dummies(time_of_day)
    df = df.drop('time_of_day', axis=1)
    df = pd.concat((df,one_hot))
    print(df)
    print(df.columns)
