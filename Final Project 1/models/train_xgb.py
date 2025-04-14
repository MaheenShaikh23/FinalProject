from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_preprocessing import load_and_split_data

os.makedirs("models", exist_ok=True)
X_train, X_test, y_train, y_test = load_and_split_data()

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)
xgb_random_search = RandomizedSearchCV(xgb, xgb_param_grid, n_iter=20, scoring='r2', cv=3, verbose=2, n_jobs=-1)
xgb_random_search.fit(X_train, y_train)

joblib.dump(xgb_random_search.best_estimator_, "models/xgboost.pkl")
print("Best XGBoost model saved.")

