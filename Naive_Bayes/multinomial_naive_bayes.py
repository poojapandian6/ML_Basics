from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

emails = [
    "Win money now",
    "Claim your free prize",
    "Meeting at 10 am",
    "Project discussion tomorrow",
    "Free lottery winner",
    "Lunch with team"
]

labels = [1, 1, 0, 0, 1, 0]

vectorizer =CountVectorizer()
X=vectorizer.fit_transform(emails)

X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.2,random_state=42)
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print(model.predict(X_test))
print(model.predict_proba(X_test))
print("Accuracy : ",accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
