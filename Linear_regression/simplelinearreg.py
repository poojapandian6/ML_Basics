import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

df=pd.read_csv("houseprice.csv")
print(df.head())

X=df[['Size']]
y=df['Price']

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

print("Slope: ",model.coef_)
print("Y intercept: ",model.intercept_)

pred=model.predict([[2600]])
print(pred)


y_pred=model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² =", r2)

mae = mean_absolute_error(y_test, y_pred)
print("MAE =", mae)

mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)

rmse = np.sqrt(mse)
print("RMSE =", rmse)

y_pred = model.predict(X)
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")

plt.show()
