from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data=load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

prediction = model.predict([X_test[0]])
print(prediction)

prob = model.predict_proba([X_test[0]])
print(prob)


print("Accuracy score: ",accuracy)

print("classification report: ",classification_report(y_test,y_pred))
