import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the root directory (where `utils` folder lives) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_preprocessing import preprocess_data

# Load models
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/meta_model.pkl", "rb") as f:
    meta_model = pickle.load(f)

# Set Streamlit page config
st.set_page_config(page_title="Waste Prediction App", layout="wide")
st.title("🗑️ Smart Waste Prediction System")

st.sidebar.header("Input Parameters")

def user_input():
    population = st.sidebar.slider("Population", 5000, 1000000, 50000)
    gdp_per_capita = st.sidebar.slider("GDP per Capita", 1000.0, 50000.0, 20000.0)
    recycling_rate = st.sidebar.slider("Recycling Rate (%)", 10.0, 90.0, 50.0)
    household_size = st.sidebar.slider("Household Size", 1.0, 10.0, 4.0)
    industrial_waste = st.sidebar.slider("Industrial Waste (kg)", 90000.0, 4500000.0, 1000000.0)
    plastic_waste = st.sidebar.slider("Plastic Waste (kg)", 50.0, 1000.0, 300.0)

    input_values = [population, gdp_per_capita, recycling_rate, household_size, industrial_waste, plastic_waste]
    columns = [
        "population", "gdp_per_capita", "recycling_rate",
        "household_size", "industrial_waste", "plastic_waste"
    ]

    input_df = pd.DataFrame([input_values], columns=columns)
    return input_df

input_df = user_input()

# Preprocess input (no scaling since scaler is removed)
input_values = np.array(input_df.iloc[0]).reshape(1, -1)

# Predict using base models
rf_pred = rf_model.predict(input_values)[0]
xgb_pred = xgb_model.predict(input_values)[0]

# Stack predictions
stacked_input = np.column_stack((rf_pred, xgb_pred))
final_pred = meta_model.predict(stacked_input)[0]

st.subheader("📈 Predicted Total Waste (kg):")
st.success(f"{final_pred:,.2f} kg")

# Display input data
with st.expander("View Input Data"):
    st.write(input_df)

# Display MAE, MSE, and R² metrics
y_test = np.array([0])  # Assuming you don't have actual data for this example
y_pred_stacked = np.array([final_pred])

mae = np.abs(y_test - y_pred_stacked).mean()  # Calculate MAE
mse = ((y_test - y_pred_stacked) ** 2).mean()  # Calculate MSE
r2 = 1 - (np.sum((y_test - y_pred_stacked) ** 2) / np.sum((y_test - y_pred_stacked.mean()) ** 2))  # Calculate R²

# Feature importance plots (Optional: remove if not needed)
st.subheader("🔍 Feature Importance")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_
features = input_df.columns

sns.barplot(x=rf_importances, y=features, ax=axes[0])
axes[0].set_title("Random Forest Feature Importance")

sns.barplot(x=xgb_importances, y=features, ax=axes[1])
axes[1].set_title("XGBoost Feature Importance")

st.pyplot(fig)

st.subheader("🔍 Model Metrics")
st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"R²: {r2:.2f}")

st.caption("Made with ❤️ using Streamlit")
