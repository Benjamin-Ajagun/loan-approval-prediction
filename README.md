# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal information.

## Project Overview

The goal of this project is to build and evaluate classification models that can assist in predicting loan approval decisions.

The project follows a complete machine learning workflow:

1. Data ingestion and initial understanding
2. Data quality investigation
3. Data cleaning
4. Exploratory Data Analysis (EDA)
5. Data splitting and preprocessing
6. Baseline model training
7. Class balance and error analysis
8. Hyperparameter tuning
9. Final model evaluation

## Dataset

The dataset contains **4,269 loan applications** and includes information such as:

- Number of dependents
- Education
- Self-employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential assets
- Commercial assets
- Luxury assets
- Bank assets
- Loan status

The target variable is:

- `Approved` — Loan application approved
- `Rejected` — Loan application rejected

`loan_id` is treated as an identifier and is excluded from model training.

## Data Preparation

During data quality checks, invalid negative values were identified in the `residential_assets_value` feature. These values were treated as missing and handled during preprocessing using median imputation.

Categorical variables were encoded using One-Hot Encoding, while numerical variables were standardized using StandardScaler.

The dataset was split using stratification:

- 70% Training
- 15% Validation
- 15% Testing

The test set was kept separate for the final evaluation.

## Models

Three classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Initial baseline models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score

Hyperparameter tuning was then performed using 5-fold cross-validation, with **F1 Score** as the optimization metric.

## Final Model

The tuned **Random Forest** model achieved the strongest validation performance and was selected as the final model.

Its tuned configuration was:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`

The final model was evaluated on the previously untouched test set.

## Evaluation

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

A confusion matrix is also used to examine the types of correct and incorrect loan predictions.

Feature importance is used to identify which variables contribute most to the Random Forest's predictions.

## Project Structure

```text
loan-approval-prediction/
│
├── loan_approval_prediction.ipynb
├── loan_approval_dataset.csv
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md