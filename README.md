# Fatigue Life Prediction using Machine Learning

This repository contains a machine learning pipeline designed to predict the fatigue life (log N) of materials based on stress and yield strength parameters. The project evaluates and compares multiple regression models to find the most accurate predictive tool.

## Machine Learning Pipeline
* **Data Preprocessing:** Handling missing values and feature scaling using `StandardScaler`.
* **Model Selection:** Implementation of Linear Regression, Ridge, and Lasso algorithms.
* **Hyperparameter Tuning:** Utilizing `GridSearchCV` to find the optimal alpha penalty values for Ridge and Lasso.
* **Validation:** 5-Fold Cross-Validation to ensure model robustness and avoid overfitting.
* **Feature Importance:** Using Permutation Importance to quantify which physical parameters impact the fatigue life most.

## Dataset Features
The model predicts `log_N` using the following parameters:
* `stress_amplitude`
* `stress_ratio`
* `yield_strength`

## How to Run
1. Ensure the dataset (`fatigue_life_data.csv`) is in the root directory.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

## Developer

Diren Gürgül - Mechanical Engineering Student at Istanbul Technical University (ITU)
