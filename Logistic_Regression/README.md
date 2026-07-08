# 📊 Logistic Regression

## 📌 Overview

Logistic Regression is a supervised machine learning algorithm used for **binary classification** problems. Instead of predicting continuous values, it predicts the probability of an instance belonging to a particular class.

The predicted probability is converted into a class label (0 or 1) using the **Sigmoid Function**.

---

## 📊 Dataset

- **Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** Scikit-learn (`load_breast_cancer()`)
- **Task:** Classify tumors as **Malignant** or **Benign**

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

---

## 📚 Concepts Covered

- Binary Classification
- Logistic Regression
- Sigmoid Function
- Log Odds (Logit)
- Maximum Likelihood Estimation
- Model Training
- Probability Prediction
- Confusion Matrix
- Accuracy Score
- Precision
- Recall
- F1-Score

---

## 📁 Files

- `logistic_regression.py`

---

## 🚀 Workflow

1. Import required libraries
2. Load the Breast Cancer dataset
3. Split the dataset into training and testing sets
4. Train the Logistic Regression model
5. Predict class labels
6. Predict class probabilities
7. Evaluate the model using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## 📈 Sample Output

- Accuracy: **96.49%**

Example Probability Prediction:

```python
Prediction: [1]

Probability:
[[0.119 0.881]]
```

Interpretation:

- Probability of Class 0 = 11.9%
- Probability of Class 1 = 88.1%
- Predicted Class = **1**

---

## 🎯 Learning Outcome

Through this implementation, I learned:

- Difference between Regression and Classification
- Why Logistic Regression is used for classification
- How the Sigmoid Function converts log-odds into probabilities
- How Logistic Regression predicts probabilities before class labels
- How to evaluate a classification model using Accuracy, Confusion Matrix, Precision, Recall, and F1-Score
- How to make predictions for new data samples

---

## 📖 Key Libraries Used

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```
