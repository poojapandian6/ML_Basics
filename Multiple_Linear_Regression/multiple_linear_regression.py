import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv("houseprice.csv")

print(df.head())

X = df[['Size', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

pred = model.predict([[2600, 4, 3]])

print("Predicted Price:", pred)

y_pred = model.predict(X_test)

print("R² =", r2_score(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)
print("MAE =", mae)

mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)

rmse = np.sqrt(mse)
print("RMSE =", rmse)
