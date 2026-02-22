# scripts/run_feature_pipeline.py

import os
import requests
from datetime import datetime
from pymongo import MongoClient

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI")
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607     # Karachi
LON = 67.0011

# ---------------- AQI CALCULATION ----------------
def calculate_aqi_pm25(pm25):
    """US EPA AQI calculation for PM2.5"""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((aqi_high - aqi_low) / (c_high - c_low))
                * (pm25 - c_low)
                + aqi_low
            )
    return None

# ---------------- MAIN ----------------
def main():
    if not MONGO_URI or not API_KEY:
        raise ValueError("Missing MONGO_URI or OPENWEATHER_API_KEY")

    # MongoDB
    client = MongoClient(MONGO_URI)
    db = client["aqi_database"]
    collection = db["aqi_features"]

    # API call
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    pm25 = data["list"][0]["components"]["pm2_5"]
    aqi = calculate_aqi_pm25(pm25)

    record = {
        "timestamp": datetime.utcnow(),
        "pm25": pm25,
        "aqi": aqi,
        "city": "Karachi"
    }

    collection.insert_one(record)

    print("✅ AQI feature data stored:", record)

if __name__ == "__main__":
    main()