import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

def preprocess_data(features):
    features = np.array(features).reshape(1, -1)
    return features

def load_and_split_data(filename="data/synthetic_waste_data.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(filename)
    X = df.drop(columns=['total_waste'])
    y = df['total_waste']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
