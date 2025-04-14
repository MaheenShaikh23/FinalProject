from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_preprocessing import load_and_split_data

os.makedirs("models", exist_ok=True)
X_train, X_test, y_train, y_test = load_and_split_data()

rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
rf_random_search = RandomizedSearchCV(rf, rf_param_grid, n_iter=20, scoring='r2', cv=3, verbose=2, n_jobs=-1)
rf_random_search.fit(X_train, y_train)

joblib.dump(rf_random_search.best_estimator_, "models/random_forest.pkl")
print("Best Random Forest model saved.")
