import pandas as pd
import numpy as np


def normalize(vector):
    mean = vector.mean()
    v_min = vector.min()
    v_max = vector.max()
    s = v_max - v_min

    return (vector - mean) / s


dataset = pd.read_csv("Housing.csv")

y_df = dataset["price"]
y = np.array([y_df]).T * 1e-6


x1_df = dataset["area"]
x2_df = dataset["bedrooms"]
x3_df = dataset["bathrooms"]

x1 = np.array(x1_df)
x2 = np.array(x2_df)
x3 = np.array(x3_df)


x1_norm = normalize(x1)
x2_norm = normalize(x2)
x3_norm = normalize(x3)

m = len(x1)
x0 = np.ones(m)

X = np.array([x0, x1_norm, x2_norm, x3_norm]).T
