# Automated Classification Pipeline with Hyperparameter Tuning

This project contains a Python function to build, tune, and evaluate classification models automatically on any tabular dataset.

## Features

- Automatically detects binary target column (with exactly 2 unique classes, no missing values).
- Separates features into categorical and numerical columns.
- Handles missing data with appropriate imputation:
  - Numerical: median imputation + scaling
  - Categorical: most frequent imputation + one-hot encoding
- Applies preprocessing pipelines with `ColumnTransformer` and `Pipeline`.
- Trains multiple classifiers:
  - Logistic Regression
  - Random Forest Classifier
  - K-Nearest Neighbors
- Uses `RandomizedSearchCV` for hyperparameter tuning with 5-fold cross-validation.
- Prints best hyperparameters and accuracy scores on both validation and test data for each model.

## Usage

1. Import necessary libraries.
2. Load your dataset as a pandas DataFrame.
3. Call the `model()` function with your dataset as input.
4. The function will output the best accuracy and parameters for each classifier, plus test accuracy.

