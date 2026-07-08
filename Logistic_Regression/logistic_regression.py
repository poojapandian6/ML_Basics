import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)
y_pred=model.predict(X_test)
print("y prediction:",y_pred[:10])
print("actual labels: ",y_test[:10])
prob=model.predict_proba(X_test)
print("prob: ",prob[:10])

print("Accuracy: ", accuracy_score(y_test,y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

patient = X_test[0].reshape(1, -1)
prediction = model.predict(patient)

probability = model.predict_proba(patient)

print(prediction)
print(probability)

