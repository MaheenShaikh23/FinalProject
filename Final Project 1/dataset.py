import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate synthetic data
data = {
    "population": np.random.randint(5000, 1000000, num_samples),  # Population in cities
    "gdp_per_capita": np.random.uniform(1000, 50000, num_samples),  # GDP per capita in $
    "recycling_rate": np.random.uniform(10, 90, num_samples),  # Recycling rate in %
    "household_size": np.random.uniform(2, 6, num_samples),  # Average household size
    "industrial_waste": np.random.uniform(90000, 4500000, num_samples),  # 
    "plastic_waste": np.random.uniform(50, 1000, num_samples)  # Plastic waste in kg
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a synthetic target variable (Total waste)
df["total_waste"] = (
    df["population"] * 0.002 +  
    df["gdp_per_capita"] * 0.01 -  
    df["recycling_rate"] * 0.05 +  
    df["household_size"] * 5 +  
    df["industrial_waste"] * 0.3 +  
    df["plastic_waste"] * 0.7 +
    np.random.normal(0, 100, num_samples)  # Add noise
)

# Save dataset
df.to_csv("data/synthetic_waste_data.csv", index=False)

print("Dataset created successfully!")
