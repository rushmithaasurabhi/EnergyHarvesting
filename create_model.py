# ------------------------------------------------------------
# Energy Prediction Model with Preprocessing and Random Forest
# ------------------------------------------------------------
# Goal: Build a machine learning pipeline to predict "Power (W)"
#       using environmental, device, or network features.
#       The pipeline includes preprocessing (cleaning, scaling,
#       imputing missing values) + Random Forest regression.
# ------------------------------------------------------------

# =========================
# 1. Import Required Libraries
# =========================

import pandas as pd                       # For data loading and manipulation
import joblib                             # To save the trained model for reuse
from sklearn.model_selection import train_test_split   # Split data into train/test
from sklearn.pipeline import Pipeline                   # Build ML pipelines
from sklearn.preprocessing import StandardScaler        # Standardize numeric features
from sklearn.impute import SimpleImputer                # Handle missing values
from sklearn.compose import ColumnTransformer           # Apply transforms to columns
from sklearn.ensemble import RandomForestRegressor      # ML algorithm

# =========================
# 2. Load Dataset
# =========================
# We assume the dataset contains device/network parameters and
# measured power usage. The target variable is "Power (W)".
# Some columns like SSID, MAC Address, and Timestamp are identifiers
# and not useful for prediction, so they are excluded from features.

df = pd.read_csv("files/Real_Data_With_Encoded_SSID.csv")

# =========================
# 3. Define Target and Features
# =========================
# y = what we want to predict → "Power (W)"
# X = input features (all other useful columns except IDs)

target_col = "Power (W)"
y = df[target_col]

# Drop columns that are identifiers or metadata not useful for modeling
X = df.drop(columns=[target_col, "SSID", "MAC Address", "Timestamp"])

# Detect column types:
# - Numerical columns (integers/floats) need scaling and imputation
# - Categorical columns (objects/strings) need separate handling
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# =========================
# 4. Build Preprocessing Pipelines
# =========================

# For NUMERIC columns:
# - Missing values: replace with median (robust against outliers)
# - Scaling: standardize values (mean=0, std=1) so ML model treats all features equally
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# For CATEGORICAL columns:
# - Missing values: replace with most frequent category
# - No encoding is added here since dataset may already have encoded values.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combine preprocessing steps:
# ColumnTransformer applies different transformations to numeric and categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# =========================
# 5. Define Machine Learning Model
# =========================
# We use RandomForestRegressor because:
# - It is a powerful ensemble algorithm (uses many decision trees).
# - Handles both numeric and categorical data well.
# - Naturally captures non-linear relationships between features.
# - Robust against outliers and missing values.
# - Provides feature importance for interpretation.
# - Parallelizable (n_jobs=-1 uses all CPU cores for speed).

model = RandomForestRegressor(
    n_estimators=300,   # number of trees in the forest
    random_state=42,    # reproducibility
    n_jobs=-1           # use all available CPU cores for faster training
)

# =========================
# 6. Create Full Pipeline
# =========================
# Why Pipeline?
# - Ensures preprocessing (imputation, scaling) is always applied
#   before passing data to the model.
# - Prevents data leakage (test data is transformed using
#   parameters learned only from training set).
# - Keeps workflow clean and reproducible.

pipe = Pipeline(steps=[
    ("pre", preprocessor),  # preprocessing step
    ("model", model)        # machine learning model
])

# =========================
# 7. Train-Test Split
# =========================
# Split data into 80% training and 20% testing
# - Training data: used to fit the model
# - Testing data: used to evaluate performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. Train the Model
# =========================
# Fit the pipeline:
# - Preprocessing is applied automatically to X_train
# - RandomForest is trained on the processed features
pipe.fit(X_train, y_train)

# =========================
# 9. Save the Model
# =========================
# Save the trained pipeline (preprocessing + model together)
# Using joblib allows us to reload it later for predictions
# without retraining.
joblib.dump(pipe, "trained_energy_model.joblib")

# ------------------------------------------------------------
# End of Script
# ------------------------------------------------------------
