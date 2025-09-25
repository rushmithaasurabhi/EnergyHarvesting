# ------------------------------------------------------------
# Energy Prediction Model Evaluation
# ------------------------------------------------------------
# Goal: Load the trained pipeline (preprocessing + RandomForest),
#       evaluate it on the test set, and report performance
#       using standard regression metrics + visualization.
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
# Use the same dataset used in training for consistency.
df = pd.read_csv("files/Real_Data_With_Encoded_SSID.csv")

# =========================
# 3. Define Target and Features
# =========================
# Target variable = Power (W), which we want to predict.
target_col = "Power (W)"
y = df[target_col]

# Drop irrelevant identifiers that do not help prediction.
X = df.drop(columns=[target_col, "SSID", "MAC Address", "Timestamp"])

# =========================
# 4. Split Dataset (Train/Test)
# =========================
# Keep the split consistent with training (80/20, random_state=42).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Load Trained Model
# =========================
# Load the previously trained and saved pipeline.
# This includes preprocessing + RandomForest model.
pipe = joblib.load("trained_energy_model.joblib")

# =========================
# 6. Generate Predictions
# =========================
# Predict on the test set to evaluate generalization performance.
y_pred = pipe.predict(X_test)

# Show a sample of actual vs predicted values for inspection.
results = pd.DataFrame({
    "Actual Power (W)": y_test.values,
    "Predicted Power (W)": y_pred
})
print(results.head(10))

# =========================
# 7. Evaluate Performance with Metrics
# =========================
# R² score (coefficient of determination): 
#   - How well the model explains variance in the data.
#   - R² = 1 means perfect prediction, 0 means no better than average.
r2 = r2_score(y_test, y_pred)

# MAE (Mean Absolute Error):
#   - Average absolute difference between predicted and actual.
#   - Easy to interpret: "on average, predictions are X watts off".
mae = mean_absolute_error(y_test, y_pred)

# RMSE (Root Mean Squared Error):
#   - Penalizes large errors more heavily than MAE.
#   - Good for sensitive applications where outliers matter.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation summary
print("\n📊 Model Evaluation on Test Data")
print("R² score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# =========================
# 8. Visualization: Predicted vs Actual
# =========================
# A scatter plot helps us see how close predictions are
# to the actual values. A perfect model would align all points
# along the 45-degree red line.

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
