"""
Synthetic Data Generator for Fatigue Life Prediction
Generates a dataset with physically sound relationships between stress, 
yield strength, and fatigue life (log N).
"""

import pandas as pd
import numpy as np

# Set the random seed for scientific reproducibility
np.random.seed(42)
n_samples = 250

# Generate features based on the expected data ranges
stress_amplitude = np.random.uniform(200, 500, n_samples) # in MPa
stress_ratio = np.random.uniform(-1, 1, n_samples)        # R ratio
yield_strength = np.random.uniform(400, 600, n_samples)   # in MPa

# Create a physically sound formula for the target variable (log_N):
# Higher stress amplitude decreases life (- coefficient)
# Higher yield strength increases life (+ coefficient)
log_N = (
    -0.018 * stress_amplitude + 
    0.4 * stress_ratio + 
    0.006 * yield_strength - 
    1.5 + 
    np.random.normal(0, 0.25, n_samples) # Adding realistic noise
)

# Create a Pandas DataFrame
df = pd.DataFrame({
    'stress_amplitude': np.round(stress_amplitude, 2),
    'stress_ratio': np.round(stress_ratio, 2),
    'yield_strength': np.round(yield_strength, 2),
    'log_N': np.round(log_N, 4)
})

# Save the generated data to a CSV file
file_name = 'fatigue_life_data.csv'
df.to_csv(file_name, index=False)

print(f"Success! '{file_name}' has been generated with {n_samples} rows of data.")
print("\nFirst 5 rows of the generated dataset:")
print(df.head())
