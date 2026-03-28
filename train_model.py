from preprocessing import load_data, prepare_features

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib
import numpy as np
import os

os.makedirs("models", exist_ok=True)


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"  {name:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "model": model}


def main():
    print("Loading data …")
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")
    print(f"  Features      : {X.shape[1]}")
    print()

    candidates = [
        ("Linear Regression",
         Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),

        ("Random Forest",
         RandomForestRegressor(n_estimators=200, max_depth=10,
                               min_samples_leaf=4, random_state=42, n_jobs=-1)),

        ("Gradient Boosting",
         GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.05, random_state=42)),
    ]

    print("Model evaluation:")
    results = [evaluate(name, mdl.fit(X_train, y_train), X_test, y_test)
               for name, mdl in candidates]

    best = min(results, key=lambda r: r["mae"])
    print(f"\nBest model  →  {best['name']}  (MAE={best['mae']:.3f})")

    joblib.dump(best["model"], "models/shipping_model.pkl")
    print("Model saved to models/shipping_model.pkl")

    # Save feature column list for simulation alignment
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    print("Feature columns saved to models/feature_columns.pkl")


if __name__ == "__main__":
    main()
