# ============================================================
# model_builder.py
# ABC Ltd. - Predictive Analytics Model Training
#
# Models:
#   1. Logistic Regression -> Employee Attrition
#   2. Linear Regression   -> Monthly Income
#
# Outputs:
#   models/logistic_model.joblib
#   models/linear_model.joblib
#   models/metrics.json
#   models/model_config.json
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "HR_Analytics.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 3. FALLBACK SYNTHETIC DATA
# ============================================================

def create_synthetic_dataset(n_rows=1500):
    """
    Creates a synthetic employee dataset so that the project
    remains runnable if the real CSV has not yet been supplied.

    For the final assignment, replace this with your actual
    open-source dataset.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    departments = [
        "Sales",
        "Research & Development",
        "Human Resources"
    ]

    business_travel = [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]

    job_roles = [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manager",
        "Human Resources",
        "Healthcare Representative"
    ]

    overtime = ["Yes", "No"]

    data = pd.DataFrame({
        "Age": rng.integers(22, 60, n_rows),
        "BusinessTravel": rng.choice(
            business_travel, n_rows, p=[0.70, 0.20, 0.10]
        ),
        "Department": rng.choice(
            departments, n_rows, p=[0.40, 0.40, 0.20]
        ),
        "DistanceFromHome": rng.integers(1, 30, n_rows),
        "JobLevel": rng.integers(1, 6, n_rows),
        "JobRole": rng.choice(job_roles, n_rows),
        "JobSatisfaction": rng.integers(1, 5, n_rows),
        "OverTime": rng.choice(
            overtime, n_rows, p=[0.72, 0.28]
        ),
        "TotalWorkingYears": rng.integers(0, 35, n_rows),
        "YearsAtCompany": rng.integers(0, 25, n_rows),
        "YearsInCurrentRole": rng.integers(0, 15, n_rows),
        "YearsSinceLastPromotion": rng.integers(0, 10, n_rows),
        "YearsWithCurrManager": rng.integers(0, 15, n_rows),
        "JobInvolvement": rng.integers(1, 5, n_rows),
        "WorkLifeBalance": rng.integers(1, 5, n_rows),
        "Education": rng.integers(1, 6, n_rows),
    })

    # Create monthly income based on several variables
    data["MonthlyIncome"] = (
        1800
        + data["JobLevel"] * 2400
        + data["TotalWorkingYears"] * 110
        + data["YearsAtCompany"] * 80
        + data["Education"] * 250
        + rng.normal(0, 900, n_rows)
    )

    data["MonthlyIncome"] = data["MonthlyIncome"].clip(lower=1200)

    # Generate attrition probability
    logit = (
        -2.2
        + 0.035 * (35 - data["Age"])
        + 0.045 * data["DistanceFromHome"]
        + 0.85 * (data["OverTime"] == "Yes").astype(int)
        - 0.45 * data["JobSatisfaction"]
        - 0.20 * data["JobInvolvement"]
        - 0.05 * data["YearsAtCompany"]
        + 0.25 * (data["BusinessTravel"] == "Travel_Frequently").astype(int)
    )

    probability = 1 / (1 + np.exp(-logit))

    data["Attrition"] = np.where(
        rng.random(n_rows) < probability,
        "Yes",
        "No"
    )

    return data


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset():
    """
    Load CSV if available.
    Otherwise generate fallback synthetic data.
    """

    if DATA_FILE.exists():
        print(f"Loading dataset: {DATA_FILE}")

        df = pd.read_csv(DATA_FILE)

        if df.empty:
            raise ValueError(
                "The CSV file exists but contains no rows."
            )

        print(f"Dataset loaded: {df.shape}")

        return df

    print("WARNING: CSV not found.")
    print("Using fallback synthetic dataset.")

    return create_synthetic_dataset()


# ============================================================
# 5. STANDARDIZE COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Removes unnecessary spaces from column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# 6. IDENTIFY TARGET COLUMNS
# ============================================================

def identify_columns(df):
    """
    Detects standard column names.
    """

    # Attrition
    attrition_candidates = [
        "Attrition",
        "Churn",
        "Churn_Flag"
    ]

    attrition_column = None

    for col in attrition_candidates:
        if col in df.columns:
            attrition_column = col
            break

    # Income
    income_candidates = [
        "MonthlyIncome",
        "Monthly_Income",
        "Salary",
        "MonthlySalary"
    ]

    income_column = None

    for col in income_candidates:
        if col in df.columns:
            income_column = col
            break

    if attrition_column is None:
        raise ValueError(
            "Could not find an Attrition/Churn target column."
        )

    if income_column is None:
        raise ValueError(
            "Could not find a MonthlyIncome/Salary target column."
        )

    return attrition_column, income_column


# ============================================================
# 7. CLEAN TARGETS
# ============================================================

def clean_targets(df, attrition_column, income_column):
    """
    Cleans target variables.
    """

    df = df.copy()

    # Clean attrition
    df[attrition_column] = (
        df[attrition_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "y": 1,
        "1": 1,
        "true": 1,
        "no": 0,
        "n": 0,
        "0": 0,
        "false": 0,
    }

    df["__attrition_target__"] = (
        df[attrition_column]
        .map(mapping)
    )

    # Convert income to numeric
    df["__income_target__"] = pd.to_numeric(
        df[income_column],
        errors="coerce"
    )

    # Remove rows where targets are unavailable
    df = df.dropna(
        subset=[
            "__attrition_target__",
            "__income_target__"
        ]
    )

    return df


# ============================================================
# 8. SELECT FEATURES
# ============================================================

def select_features(df, attrition_column, income_column):
    """
    Uses a compact, manager-friendly feature set.

    These features are deliberately limited so that managers
    do not have to enter 30+ variables in the Streamlit app.
    """

    preferred_features = [
        "Age",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "OverTime",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "JobInvolvement",
        "WorkLifeBalance",
        "Education",
    ]

    available_features = [
        col for col in preferred_features
        if col in df.columns
    ]

    if len(available_features) < 5:
        raise ValueError(
            "Too few expected feature columns were found. "
            "Please check your dataset."
        )

    return available_features


# ============================================================
# 9. CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(X):
    """
    Creates a robust preprocessing pipeline.

    Numeric:
        Missing values -> median
        Standardization -> Z-score

    Categorical:
        Missing values -> most frequent
        One-hot encoding
    """

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# 10. TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_model(
    df,
    feature_columns
):
    """
    Train employee attrition classification model.
    """

    X = df[feature_columns].copy()
    y = df["__attrition_target__"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = create_preprocessor(X_train)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "test_size": int(len(y_test)),
        "positive_cases": int(y_test.sum()),
    }

    return model, metrics


# ============================================================
# 11. TRAIN LINEAR REGRESSION
# ============================================================

def train_linear_model(
    df,
    feature_columns
):
    """
    Train monthly income regression model.
    """

    X = df[feature_columns].copy()
    y = df["__income_target__"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    preprocessor = create_preprocessor(X_train)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    metrics = {
        "r2_score": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "test_size": int(len(y_test)),
    }

    return model, metrics


# ============================================================
# 12. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("ABC LTD. PREDICTIVE ANALYTICS MODEL BUILDER")
    print("=" * 70)

    # Load
    df = load_dataset()

    # Clean columns
    df = clean_column_names(df)

    print("\nColumns detected:")
    print(df.columns.tolist())

    # Identify targets
    attrition_column, income_column = identify_columns(
        df
    )

    print(
        f"\nAttrition target: {attrition_column}"
    )

    print(
        f"Income target: {income_column}"
    )

    # Clean targets
    df = clean_targets(
        df,
        attrition_column,
        income_column
    )

    # Feature selection
    feature_columns = select_features(
        df,
        attrition_column,
        income_column
    )

    print("\nFeatures used:")
    for feature in feature_columns:
        print(f"  - {feature}")

    # Train logistic
    print("\nTraining Logistic Regression...")

    logistic_model, logistic_metrics = train_logistic_model(
        df,
        feature_columns
    )

    # Train linear
    print("\nTraining Linear Regression...")

    linear_model, linear_metrics = train_linear_model(
        df,
        feature_columns
    )

    # Save models
    logistic_path = MODEL_DIR / "logistic_model.joblib"
    linear_path = MODEL_DIR / "linear_model.joblib"

    joblib.dump(
        logistic_model,
        logistic_path
    )

    joblib.dump(
        linear_model,
        linear_path
    )

    # Save metrics
    metrics = {
        "logistic_regression": logistic_metrics,
        "linear_regression": linear_metrics,
    }

    with open(
        MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save configuration
    config = {
        "feature_columns": feature_columns,
        "attrition_column": attrition_column,
        "income_column": income_column,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    with open(
        MODEL_DIR / "model_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # Print results
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 70)

    for key, value in logistic_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION RESULTS")
    print("=" * 70)

    for key, value in linear_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(logistic_path)
    print(linear_path)
    print(MODEL_DIR / "metrics.json")
    print(MODEL_DIR / "model_config.json")

    print("\nModel training completed successfully.")


# ============================================================
# 13. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
