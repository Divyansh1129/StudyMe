import os
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import {
    mean_absolute_error,
    mean_squared_error,
    r2_score
}

data = pd.read_csv("dataset/Students Performance Dataset.csv")

X = data[["Study_Hours_per_Week"]]
y = data["Final_Score"]




