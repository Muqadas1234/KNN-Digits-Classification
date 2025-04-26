# KNN Digits Classification

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify handwritten digits from the popular `digits` dataset available in scikit-learn. It includes:

- Loading and exploring the dataset
- Splitting the data into training and testing sets
- Training KNN models for different values of K
- Finding the best value of K based on accuracy
- Evaluating the final model using:
  - Accuracy score
  - Confusion matrix
  - Classification report
- Checking for overfitting or underfitting
- Visualizing performance (K vs Accuracy graph and Confusion Matrix)

## Libraries Used
- `scikit-learn`
- `matplotlib`
- `numpy` (indirectly via scikit-learn)

## How to Run
1. Make sure you have Python installed.
2. Install required libraries if not already installed:
   ```bash
   pip install scikit-learn matplotlib
