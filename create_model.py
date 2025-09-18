import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# 1. Load your dataset
# ----------------------------
df = pd.read_csv("files/Real_Data_With_Encoded_SSID.csv")

# ----------------------------
# 2. Define target and features
# ----------------------------
target_col = "Power (W)"      # what we want to predict
y = df[target_col]

# drop columns that are IDs or not useful as predictors
X = df.drop(columns=[target_col, "SSID", "MAC Address", "Timestamp"])

# detect column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ----------------------------
# 3. Build preprocessing
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ----------------------------
# 4. Define model
# ----------------------------
model = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1
)

# Full pipeline = preprocessing + model
pipe = Pipeline(steps=[("pre", preprocessor),
                       ("model", model)])

# ----------------------------
# 5. Train-test split (80/20)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6. Train the model
# ----------------------------
pipe.fit(X_train, y_train)

# ----------------------------
# 7. Save as joblib
# ----------------------------
joblib.dump(pipe, "trained_energy_model.joblib")
