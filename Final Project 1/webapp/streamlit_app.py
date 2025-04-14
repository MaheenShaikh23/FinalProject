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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load test data to evaluate model (you'll need to import this function)
from utils.data_preprocessing import load_and_split_data
_, X_test, _, y_test = load_and_split_data()

# Predict using trained base models
rf_test_pred = rf_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

# Stack test predictions
stacked_test = np.column_stack((rf_test_pred, xgb_test_pred))
stacked_pred = meta_model.predict(stacked_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, stacked_pred)
mse = mean_squared_error(y_test, stacked_pred)
r2 = r2_score(y_test, stacked_pred)

# Sensitivity Analysis: Effect of Each Input on Predicted Waste
st.subheader("📊 Effect of Each Input on Total Waste")

# Use current user input as baseline
baseline = input_df.iloc[0].copy()

# Ranges for each feature to test
ranges = {
    "population": np.linspace(5000, 1000000, 20),
    "gdp_per_capita": np.linspace(1000, 50000, 20),
    "recycling_rate": np.linspace(10, 90, 20),
    "household_size": np.linspace(1, 10, 20),
    "industrial_waste": np.linspace(90000, 4500000, 20),
    "plastic_waste": np.linspace(50, 1000, 20)
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (feature, values) in enumerate(ranges.items()):
    preds = []
    for val in values:
        temp_input = baseline.copy()
        temp_input[feature] = val
        temp_df = pd.DataFrame([temp_input])
        
        rf_pred = rf_model.predict(temp_df)[0]
        xgb_pred = xgb_model.predict(temp_df)[0]
        stacked_input = np.column_stack((rf_pred, xgb_pred))
        final_pred = meta_model.predict(stacked_input)[0]
        
        preds.append(final_pred)
    
    axes[i].plot(values, preds, marker='o')
    axes[i].set_title(f"Impact of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Predicted Total Waste")

plt.tight_layout()
st.pyplot(fig)

st.subheader("📏 Model Evaluation Metrics")
st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"R²: {r2:.2f}")

st.caption("Made with ❤️ using Streamlit")
