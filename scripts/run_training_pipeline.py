# scripts/run_training_pipeline.py

import os
import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI")
MODEL_DIR = "models"

# ---------------- MAIN ----------------
def main():
    if not MONGO_URI:
        raise ValueError("Missing MONGO_URI")

    # MongoDB
    client = MongoClient(MONGO_URI)
    db = client["aqi_database"]
    collection = db["aqi_features"]

    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("No training data found")

    X = df[["pm25"]]
    y = df["aqi"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"📊 Model MAE: {mae:.2f}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("✅ Model and scaler saved")

if __name__ == "__main__":
    main()