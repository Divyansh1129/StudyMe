import os
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

data = pd.read_csv("dataset/Students Performance Dataset.csv")

X = data[["Study_Hours_per_Week"]]
y = data["Final_Score"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("Model Performance")
print("=" * 40)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print("=" * 40)

os.makedirs("models", exist_ok=True)

with open("models/linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("Location: models/linear_model.pkl")


