"""
Fatigue Life Prediction using Machine Learning
Models: Linear Regression, Ridge Regression, Lasso Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    """Loads the dataset and handles missing values."""
    df = pd.read_csv(file_path)
    if df.isnull().sum().any():
        df = df.dropna()
        print("Dataframe has been cleaned of missing values.")
    return df

def optimize_hyperparameters(X_train, y_train):
    """Performs hyperparameter optimization for Ridge and Lasso regression."""
    # Ridge Optimization
    ridge_params = {"alpha": np.logspace(-3, 3, 10)}
    ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="r2")
    ridge_cv.fit(X_train, y_train)

    # Lasso Optimization
    lasso_params = {"alpha": np.logspace(-4, 2, 10)}
    lasso_cv = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring="r2")
    lasso_cv.fit(X_train, y_train)

    print(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']:.4f}")
    print(f"Best alpha for Lasso: {lasso_cv.best_params_['alpha']:.4f}\n")

    return ridge_cv.best_estimator_, lasso_cv.best_estimator_

def plot_model_performance(results):
    """Compares and plots the R² and RMSE scores of the evaluated models."""
    metrics_df = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R2 Score Plot
    sns.barplot(x=metrics_df.index, y='R²', data=metrics_df, ax=axes[0], hue=metrics_df.index, palette='viridis', legend=False)
    axes[0].set_title('Model Comparison (R² Score)')
    axes[0].set_ylim(0, 1)

    # RMSE Score Plot
    sns.barplot(x=metrics_df.index, y='RMSE', data=metrics_df, ax=axes[1], hue=metrics_df.index, palette='magma', legend=False)
    axes[1].set_title('Model Comparison (RMSE)')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Data Preparation
    df = load_data("fatigue_life_data.csv")
    X = df.drop('log_N', axis=1)
    y = df['log_N']

    # Splitting the data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Optimization and Training
    best_ridge, best_lasso = optimize_hyperparameters(X_train_scaled, y_train)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": best_ridge,
        "Lasso": best_lasso
    }

    results = {}
    print("--- Model Evaluation on Test Set ---")
    for name, model in models.items():
        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        print(f"{name} evaluation completed.")

    # 4. Visualization
    plot_model_performance(results)

if __name__ == "__main__":
    main()
