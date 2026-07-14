# K-Nearest Neighbors (KNN) Classification

## 📌 Overview

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm using Scikit-learn for a binary classification problem. KNN is a **supervised, non-parametric, instance-based (lazy learning)** algorithm that classifies a new data point based on the majority class of its K nearest neighbors.

The project uses the **Breast Cancer Wisconsin Dataset** and includes feature scaling, model training, prediction, and performance evaluation.

---

## 📂 Dataset

- **Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Task:** Binary Classification
- **Target Classes:**
  - 0 → Malignant
  - 1 → Benign

---

## 🚀 Workflow

1. Load the dataset
2. Split the dataset into training and testing sets
3. Standardize features using **StandardScaler**
4. Create the KNN model
5. Train the model
6. Predict test data
7. Evaluate model performance

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn

---

## 📚 Machine Learning Concepts Covered

- K-Nearest Neighbors (KNN)
- Lazy Learning
- Euclidean Distance
- Choosing the value of K
- Majority Voting
- Weighted KNN (Concept)
- Feature Scaling (Standardization)
- KNN Classification
- KNN Regression (Concept)
- Bias-Variance Tradeoff
- Time Complexity
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## 📊 Sample Output

```
Accuracy: 94.74%

Confusion Matrix

[[40  3]
 [ 3 68]]

Classification Report

Precision : 0.93 / 0.96
Recall    : 0.93 / 0.96
F1-Score  : 0.93 / 0.96
```

---

## 💡 Key Learning Outcomes

- Understood how KNN predicts using the nearest neighbors.
- Learned why feature scaling is essential for distance-based algorithms.
- Explored different distance metrics such as Euclidean, Manhattan, Minkowski, Hamming, and Cosine Distance.
- Understood the importance of selecting an appropriate value for K.
- Learned the bias-variance tradeoff in KNN.
- Implemented KNN Classification using Scikit-learn.
- Evaluated the model using multiple performance metrics.

---

## 📌 Conclusion

KNN is a simple yet powerful machine learning algorithm that makes predictions based on the similarity between data points. Although training is fast because it simply stores the training data, prediction can be computationally expensive since distances to all training samples must be calculated.

This project provides a complete implementation of the KNN classification workflow using Python and Scikit-learn.
