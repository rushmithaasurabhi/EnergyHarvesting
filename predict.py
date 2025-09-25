# ------------------------------------------------------------
# Energy Prediction Model Evaluation + Feature Importance
# ------------------------------------------------------------
# Goal:
#  - Evaluate trained pipeline on test data
#  - Report performance metrics (R², MAE, RMSE)
#  - Visualize predictions vs actual values
#  - Analyze feature importance from Random Forest
# ------------------------------------------------------------

# =========================
# 1. Import Required Libraries
# =========================
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("files/Real_Data_With_Encoded_SSID.csv")

# =========================
# 3. Define Target and Features
# =========================
target_col = "Power (W)"
y = df[target_col]
X = df.drop(columns=[target_col, "SSID", "MAC Address", "Timestamp"])

# Keep track of feature names for later (needed for feature importance)
feature_names = X.columns.tolist()

# =========================
# 4. Train-Test Split
# =========================
# Ensure same split as during training for consistency
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Load Trained Pipeline
# =========================
# This pipeline includes:
#  - Preprocessing (imputation, scaling, categorical handling)
#  - RandomForestRegressor model
pipe = joblib.load("trained_energy_model.joblib")

# =========================
# 6. Predict on Test Set
# =========================
y_pred = pipe.predict(X_test)

# Show sample predictions for comparison
results = pd.DataFrame({
    "Actual Power (W)": y_test.values,
    "Predicted Power (W)": y_pred
})
print("\n🔍 Sample Predictions:")
print(results.head(10))

# =========================
# 7. Performance Evaluation
# =========================
# R² Score = proportion of variance explained by model
r2 = r2_score(y_test, y_pred)

# MAE = average absolute error in predictions
mae = mean_absolute_error(y_test, y_pred)

# RMSE = root of mean squared error, penalizes large errors more
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation summary
print("\n📊 Model Evaluation on Test Data")
print("R² score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# =========================
# 8. Visualization: Predicted vs Actual
# =========================
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, s=20, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.title("Predicted vs Actual Power on Test Data")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 9. Feature Importance Analysis
# =========================
# Why? Random Forest provides "feature importance" values that
# measure how much each feature contributes to reducing error
# across all trees. This helps explain the model.

# Extract trained RandomForest model from pipeline
rf_model = pipe.named_steps["model"]

# Get importance scores
importances = rf_model.feature_importances_

# Create DataFrame for ranking
feat_importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🌟 Feature Importance Ranking:")
print(feat_importances)

# =========================
# 10. Visualization: Feature Importances
# =========================
plt.figure(figsize=(10, 6))
plt.barh(feat_importances["Feature"], feat_importances["Importance"], color="teal")
plt.gca().invert_yaxis()  # highest importance at top
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.show()

# ------------------------------------------------------------
# End of Script
# ------------------------------------------------------------
