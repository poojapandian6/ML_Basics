import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler =StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
