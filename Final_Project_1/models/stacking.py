import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_preprocessing import load_and_split_data

X_train, X_test, y_train, y_test = load_and_split_data()

with open("models/random_forest.pkl", "rb") as f:
    rf_best = pickle.load(f)

with open("models/xgboost.pkl", "rb") as f:
    xgb_best = pickle.load(f)

# Get predictions from base models
train_pred_rf = rf_best.predict(X_train)
train_pred_xgb = xgb_best.predict(X_train)
test_pred_rf = rf_best.predict(X_test)
test_pred_xgb = xgb_best.predict(X_test)

# Stack predictions
stacked_train = np.column_stack((train_pred_rf, train_pred_xgb))
stacked_test = np.column_stack((test_pred_rf, test_pred_xgb))

# Train the meta-model
meta_model = LinearRegression()
meta_model.fit(stacked_train, y_train)

# Save the stacked model
with open("models/meta_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)

print("Stacked meta-model saved.")

# Predict with the stacked model
y_pred_stacked = meta_model.predict(stacked_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred_stacked)
mse = mean_squared_error(y_test, y_pred_stacked)
r2 = r2_score(y_test, y_pred_stacked)

print(f"Tuned Stacked Model - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")
