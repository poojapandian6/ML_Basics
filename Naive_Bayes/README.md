# Naive Bayes

Naive Bayes is a probabilistic supervised machine learning algorithm based on **Bayes' Theorem**. It is commonly used for classification tasks such as spam detection, sentiment analysis, document classification, and medical diagnosis.

## Algorithms Implemented

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

## Concepts Covered

- Bayes' Theorem
- Prior Probability
- Likelihood
- Posterior Probability
- Probability Density Function (Gaussian Distribution)
- Feature Independence Assumption
- Laplace Smoothing
- Log Probabilities
- Text Classification

## Implementations

### Gaussian Naive Bayes
Used for **continuous numerical features**.

Examples:
- Medical diagnosis
- Iris flower classification
- Breast cancer prediction

Stores:
- Mean (μ)
- Standard Deviation (σ)
- Prior Probability

During prediction:
- Computes Gaussian Probability Density (Likelihood)
- Calculates Posterior Probability
- Predicts the class with the highest posterior probability

---

### Multinomial Naive Bayes
Used for **count-based features**, especially text data.

Examples:
- Spam Detection
- News Classification
- Sentiment Analysis

Stores:
- Word Frequencies
- Likelihood Probabilities
- Prior Probability

During prediction:
- Looks up stored likelihoods
- Computes posterior probabilities
- Predicts the most probable class

---

### Bernoulli Naive Bayes
Used for **binary (0/1) features**.

Examples:
- Email Spam Detection (Word Present/Absent)
- Medical Symptoms (Yes/No)
- Loan Approval Features

Stores:
- Binary Feature Probabilities
- Prior Probability

During prediction:
- Considers both feature presence and absence
- Computes posterior probabilities
- Predicts the most probable class

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Files

```
gaussian_naive_bayes.py
multinomial_naive_bayes.py
bernoulli_naive_bayes.py
```

## Applications

- Spam Email Detection
- Sentiment Analysis
- News Classification
- Medical Diagnosis
- Document Classification
- Recommendation Systems

---

### Learning Outcome

After completing these implementations, you will understand:

- Bayes' Theorem
- Prior, Likelihood and Posterior
- Gaussian Probability Density Function
- Laplace Smoothing
- Difference between Gaussian, Multinomial and Bernoulli Naive Bayes
- Practical implementation using Scikit-learn
