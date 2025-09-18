import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
df = pd.read_csv("files/Real_Data_With_Encoded_SSID.csv")

# 2. Define target and features
target_col = "Power (W)"
y = df[target_col]
X = df.drop(columns=[target_col, "SSID", "MAC Address", "Timestamp"])

# 3. Train-test split (80/20 with same random_state as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Load trained model
pipe = joblib.load("trained_energy_model.joblib")

# 5. Predict on test set
y_pred = pipe.predict(X_test)


results = pd.DataFrame({
    "Actual Power (W)": y_test.values,
    "Predicted Power (W)": y_pred
})
print(results.head(10))

# 6. Evaluate performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # <-- fixed

print("📊 Model Evaluation on Test Data")
print("R² score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# 7. Plot predicted vs actual
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, s=20, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linewidth=2)  # 45-degree line
plt.xlabel("Actual Power (W)")
plt.ylabel("Predicted Power (W)")
plt.title("Predicted vs Actual Power on Test Data")
plt.grid(True)
plt.show()
