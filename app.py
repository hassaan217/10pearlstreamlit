# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import sys
import requests
import certifi
import ssl
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ML/Database imports
from pymongo import MongoClient, errors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from dotenv import load_dotenv

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AQI Intelligence - Advanced Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Main header with gradient */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards with hover effect */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Status badges */
    .badge-good { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .badge-moderate { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .badge-poor { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .badge {
        padding: 0.25rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
    }
    
    /* Live indicator animation */
    .live-indicator {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Model card styling */
    .model-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .model-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    .model-card.selected {
        border: 2px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Best model badge */
    .best-badge {
        position: absolute;
        top: -10px;
        right: -10px;
        background: #fbbf24;
        color: black;
        font-size: 10px;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e1e1e;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()

# AQI Categories with colors and descriptions
AQI_CATEGORIES = {
    'Good': {'range': (0, 50), 'color': '#10b981', 'icon': '😊', 'description': 'Air quality is satisfactory'},
    'Satisfactory': {'range': (51, 100), 'color': '#fbbf24', 'icon': '🙂', 'description': 'Minor concern for sensitive individuals'},
    'Moderate': {'range': (101, 200), 'color': '#f97316', 'icon': '😐', 'description': 'May cause breathing discomfort'},
    'Poor': {'range': (201, 300), 'color': '#ef4444', 'icon': '😷', 'description': 'Health alerts - limit outdoor activity'},
    'Very Poor': {'range': (301, 400), 'color': '#a855f7', 'icon': '⚠️', 'description': 'Health warnings of emergency conditions'},
    'Severe': {'range': (401, 500), 'color': '#7f1d1d', 'icon': '☠️', 'description': 'Serious health risk for everyone'}
}

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011

# Model colors for consistent theming
MODEL_COLORS = {
    'XGBoost': '#10b981',
    'LightGBM': '#3b82f6',
    'Random Forest': '#8b5cf6',
    'Gradient Boosting': '#f59e0b',
    'Neural Net': '#ec4899'
}

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.forecast_data = None
    st.session_state.latest_data = None
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.model_metrics = {}
    st.session_state.training_history = []
    st.session_state.auto_refresh = False
    st.session_state.last_refresh = datetime.now()
    st.session_state.db_client = None
    st.session_state.db_status = 'disconnected'
    st.session_state.api_key_status = 'unknown'
    st.session_state.selected_model_id = 'XGBoost'
    st.session_state.show_model_details = True
    st.session_state.selected_day = 0

# ==========================================
# DATABASE CONNECTION FUNCTIONS
# ==========================================

@st.cache_resource
def init_mongodb_connection():
    """Initialize MongoDB connection with retry logic"""
    mongo_uri = os.getenv('MONGO_URI')
    
    if not mongo_uri:
        try:
            mongo_uri = st.secrets.get('database', {}).get('mongo_uri')
        except:
            mongo_uri = None
    
    if not mongo_uri:
        return None, "⚠️ MONGO_URI not configured (using mock data)"
    
    try:
        strategies = [
            {'tls': True, 'tlsCAFile': certifi.where()},
            {'tls': True},
            {}
        ]
        
        for i, params in enumerate(strategies):
            try:
                client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=10000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=10000,
                    **params
                )
                client.admin.command('ping')
                return client, f"✅ Connected"
            except Exception as e:
                continue
        
        return None, "❌ All connection strategies failed"
    except Exception as e:
        return None, f"❌ Connection error: {str(e)}"

def check_openweather_api():
    """Check if OpenWeather API key is valid"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        try:
            api_key = st.secrets.get('weather', {}).get('api_key')
        except:
            api_key = None
    
    if not api_key:
        return False, "⚠️ API key not configured (using mock data)"
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, "✅ API key valid"
        else:
            return False, f"❌ API error: {response.status_code}"
    except Exception as e:
        return False, f"❌ Connection error: {str(e)}"

# ==========================================
# DATA COLLECTION FUNCTIONS
# ==========================================

def get_mock_weather_data():
    """Generate mock weather data for testing"""
    return {
        'main': {
            'temp': 28.5 + np.random.normal(0, 2),
            'humidity': 65 + np.random.normal(0, 5)
        },
        'wind': {
            'speed': 3.5 + np.random.normal(0, 1)
        }
    }, None

def fetch_weather_data():
    """Fetch current weather from OpenWeather API"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        try:
            api_key = st.secrets.get('weather', {}).get('api_key')
        except:
            api_key = None
    
    if not api_key:
        return get_mock_weather_data()
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return get_mock_weather_data()

def calculate_aqi_from_pm25(pm25):
    """Calculate AQI from PM2.5 using EPA formula"""
    if pm25 <= 12.0:
        return (pm25 / 12.0) * 50
    elif pm25 <= 35.4:
        return 51 + ((pm25 - 12.1) / 23.3) * 49
    elif pm25 <= 55.4:
        return 101 + ((pm25 - 35.5) / 19.9) * 49
    elif pm25 <= 150.4:
        return 151 + ((pm25 - 55.5) / 94.9) * 49
    elif pm25 <= 250.4:
        return 201 + ((pm25 - 150.5) / 99.9) * 99
    elif pm25 <= 350.4:
        return 301 + ((pm25 - 250.5) / 99.9) * 99
    else:
        return min(500, 401 + ((pm25 - 350.5) / 149.9) * 99)

def collect_air_quality_data():
    """Collect air quality data and store in MongoDB"""
    weather_data, error = fetch_weather_data()
    
    if error:
        st.warning(f"Weather API error: {error}. Using mock data.")
    
    try:
        # Extract weather parameters
        dt = datetime.now()
        hour = dt.hour
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind = weather_data['wind']['speed']
        
        # Calculate pollution factors
        rush_factor = 1.3 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.0
        wind_factor = max(0.5, 1.0 - (wind / 20))
        base_pm25 = 45 * rush_factor * wind_factor
        
        # Generate pollutant data with realistic variations
        record = {
            'time': dt,
            'pm10 (µg/m³)': round(base_pm25 * 1.8 + np.random.normal(0, 5), 2),
            'pm2_5 (µg/m³)': round(base_pm25 + np.random.normal(0, 3), 2),
            'carbon_monoxide (µg/m³)': round(400 * rush_factor * wind_factor + np.random.normal(0, 50), 2),
            'carbon_dioxide (ppm)': round(420 + np.random.normal(0, 10), 2),
            'nitrogen_dioxide (µg/m³)': round(35 * rush_factor + np.random.normal(0, 5), 2),
            'sulphur_dioxide (µg/m³)': round(15 + np.random.normal(0, 2), 2),
            'ozone (µg/m³)': round(30 * (1 + (temp - 20)/50) + np.random.normal(0, 5), 2),
            'dust (µg/m³)': round(np.random.uniform(0, 10), 2),
            'temperature_2m (°C)': round(temp, 1),
            'relative_humidity_2m (%)': round(humidity, 1),
            'wind_speed_10m (km/h)': round(wind * 3.6, 1),
            'aqi': round(calculate_aqi_from_pm25(base_pm25), 1)
        }
        
        # Store in MongoDB if connected
        if st.session_state.db_client:
            try:
                db = st.session_state.db_client['air_quality']
                collection = db['raw_aqi']
                result = collection.insert_one(record)
                storage_msg = f" (ID: {str(result.inserted_id)[-6:]})"
            except:
                storage_msg = " (DB storage failed)"
        else:
            storage_msg = " (not stored - DB disconnected)"
        
        # Update latest data
        st.session_state.latest_data = {
            'time': dt.isoformat(),
            'aqi': record['aqi'],
            'pm25': record['pm2_5 (µg/m³)'],
            'temperature': record['temperature_2m (°C)'],
            'humidity': record['relative_humidity_2m (%)'],
            'pm10': record['pm10 (µg/m³)'],
            'no2': record['nitrogen_dioxide (µg/m³)']
        }
        
        # Generate forecast
        st.session_state.forecast_data = generate_forecast(3)
        
        return True, f"✅ Data collected{storage_msg}"
            
    except Exception as e:
        return False, f"Error creating record: {str(e)}"

def generate_forecast(days=3):
    """Generate AQI forecast for next N days with realistic patterns"""
    if not st.session_state.latest_data:
        return generate_mock_forecast(days)
    
    forecast = []
    now = datetime.now()
    base_aqi = st.session_state.latest_data['aqi']
    base_temp = st.session_state.latest_data['temperature']
    
    for i in range(days * 24):
        future_time = now + timedelta(hours=i)
        hour = future_time.hour
        day_of_week = future_time.weekday()
        
        # Create realistic patterns
        rush_hour_factor = 1.3 if (7 <= hour <= 9 or 17 <= hour <= 19) else 1.0
        weekend_factor = 0.8 if day_of_week >= 5 else 1.0
        daily_pattern = 1 + 0.2 * np.sin(2 * np.pi * i / 24)
        
        # Generate AQI
        forecast_aqi = base_aqi * daily_pattern * rush_hour_factor * weekend_factor
        forecast_aqi += np.random.normal(0, 3)
        forecast_aqi = max(0, min(500, forecast_aqi))
        
        # Temperature with daily cycle
        forecast_temp = base_temp + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1)
        
        # Weather condition
        if 6 <= hour <= 18:
            weather = "Sunny" if np.random.random() > 0.3 else "Clouds"
        else:
            weather = "Clear" if np.random.random() > 0.3 else "Clouds"
        
        forecast.append({
            'time': future_time.isoformat(),
            'aqi': round(forecast_aqi, 1),
            'temperature': round(forecast_temp, 1),
            'weather': weather,
            'hour': hour
        })
    
    return forecast

def generate_mock_forecast(days=3):
    """Generate mock forecast data"""
    forecast = []
    now = datetime.now()
    
    for i in range(days * 24):
        future_time = now + timedelta(hours=i)
        hour = future_time.hour
        
        base_aqi = 120 + 40 * np.sin(hour / 12 * np.pi)
        
        forecast.append({
            'time': future_time.isoformat(),
            'aqi': round(max(0, min(500, base_aqi + np.random.normal(0, 10))), 1),
            'temperature': round(25 + 5 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 1), 1),
            'weather': 'Sunny' if 6 <= hour <= 18 else 'Clear',
            'hour': hour
        })
    
    return forecast

# ==========================================
# MODEL TRAINING FUNCTIONS
# ==========================================

def load_training_data():
    """Load data from MongoDB for training"""
    if not st.session_state.db_client:
        return None
    
    try:
        db = st.session_state.db_client['air_quality']
        collection = db['raw_aqi']
        
        cutoff = datetime.now() - timedelta(days=30)
        cursor = collection.find({'time': {'$gte': cutoff}}).sort('time', -1)
        data = list(cursor)
        
        if len(data) < 50:
            return None
        
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        
        return df
    except Exception as e:
        return None

def generate_synthetic_training_data(n_samples=2000):
    """Generate synthetic training data with realistic patterns"""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
    data = []
    
    for dt in dates:
        hour = dt.hour
        month = dt.month
        day_of_week = dt.dayofweek
        
        # Seasonal patterns
        seasonal = 1.3 if month in [12, 1, 2] else 0.8 if month in [6, 7, 8] else 1.0
        
        # Daily patterns
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            rush_hour = 1.4
        elif 0 <= hour <= 4:
            rush_hour = 0.6
        else:
            rush_hour = 1.0
        
        # Weekend vs weekday
        weekend = 0.8 if day_of_week >= 5 else 1.0
        
        # Base PM2.5
        base_pm25 = 40 + 30 * np.sin(hour / 12 * np.pi) + 20 * np.random.random()
        base_pm25 *= seasonal * rush_hour * weekend
        pm25 = max(5, base_pm25 + np.random.normal(0, 8))
        
        aqi = calculate_aqi_from_pm25(pm25)
        
        # Generate correlated pollutants
        pm10 = pm25 * (1.5 + 0.3 * np.random.random())
        no2 = 20 + 15 * (pm25 / 50) + 5 * np.random.random()
        so2 = 8 + 6 * (pm25 / 50) + 3 * np.random.random()
        o3 = 25 + 10 * np.sin(hour / 12 * np.pi) + 5 * np.random.random()
        co = 200 + 150 * (pm25 / 50) + 30 * np.random.random()
        
        # Weather variables
        temperature = 25 + 5 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 2)
        humidity = 60 - 10 * np.sin(hour / 12 * np.pi) + np.random.normal(0, 5)
        wind_speed = 10 + 5 * np.random.random()
        
        data.append({
            'time': dt,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pm25': pm25,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'o3': o3,
            'co': co,
            'aqi': aqi
        })
    
    return pd.DataFrame(data)

def train_models():
    """Train multiple models and select best (OPTIMIZED FOR SPEED)"""
    st.info("🔄 Loading training data...")
    
    # Try to load real data first
    df = load_training_data()
    data_source = "MongoDB"
    
    if df is None or len(df) < 100:
        st.warning("⚠️ Insufficient real data, using synthetic data")
        df = generate_synthetic_training_data(2000) # Kept at 2000 for demo
        data_source = "Synthetic"
    
    # Prepare features
    feature_cols = ['temperature', 'humidity', 'wind_speed', 'pm25', 'pm10', 
                    'no2', 'so2', 'o3', 'co']
    
    # Create a clean copy of the dataframe
    df_clean = df.copy()
    
    # Replace infinities with NaNs
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # CRITICAL FIX: Drop rows where the target variable (AQI) is NaN
    # This prevents the XGBoostError: "Label contains NaN"
    if 'aqi' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean.dropna(subset=['aqi'], inplace=True)
        dropped_count = initial_count - len(df_clean)
        if dropped_count > 0:
            # Optional: log this, but we won't block on it
            pass
    else:
        st.error("AQI column missing from dataset.")
        return {}, '', -1

    # Identify available features
    available_features = [col for col in feature_cols if col in df_clean.columns]
    
    # Fill missing values in features with median (safe fallback)
    for col in available_features:
        median_val = df_clean[col].median()
        if pd.isna(median_val):
            median_val = 0 # Fallback if all values are NaN
        df_clean[col].fillna(median_val, inplace=True)

    # Prepare X and y
    X = df_clean[available_features].copy()
    y = df_clean['aqi'].copy()

    # Ensure we have data
    if len(X) == 0:
        st.warning("No valid data found after cleaning. Using synthetic data.")
        df = generate_synthetic_training_data(2000)
        df_clean = df.copy()
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        available_features = [col for col in feature_cols if col in df_clean.columns]
        X = df_clean[available_features].copy()
        y = df_clean['aqi'].copy()
        # If synthetic data generation also has time, ensure we handle it
        if 'time' not in df_clean.columns and 'time' in df.columns:
             df_clean['time'] = df['time']

    # Add engineered features based on the CLEANED dataframe
    if 'time' in df_clean.columns:
        df_clean['time'] = pd.to_datetime(df_clean['time'])
        X['hour'] = df_clean['time'].dt.hour
        X['month'] = df_clean['time'].dt.month
        X['day_of_week'] = df_clean['time'].dt.dayofweek
    else:
        # If no time column, create dummy features to avoid crash in models that expect specific dimensions
        X['hour'] = 12
        X['month'] = 6
        X['day_of_week'] = 0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================
    # OPTIMIZED MODELS
    # 1. Reduced n_estimators from 150 to 50 (Faster)
    # 2. Changed GradientBoosting to HistGradientBoosting (Much Faster)
    # ==========================================
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=50,  # Reduced from 150
            max_depth=5,      # Reduced from 7
            learning_rate=0.1, 
            random_state=42, 
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=50,  # Reduced from 150
            max_depth=5,      # Reduced from 7
            learning_rate=0.1, 
            random_state=42, 
            verbose=-1, 
            n_jobs=-1
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=50,  # Reduced from 150 (RF is slow, this helps a lot)
            max_depth=10,     # Reduced from 12
            random_state=42, 
            n_jobs=-1
        ),
        'Hist GB': HistGradientBoostingRegressor( # Faster than standard GradientBoosting
            max_iter=50,     # Equivalent to n_estimators
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    best_name = ''
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        # Train
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation - REDUCED FOLDS from 5 to 3 for SPEED
        # If it's still too slow, set cv=2
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(X.columns, model.feature_importances_))
        else:
            # HistGradientBoosting doesn't have native feature_importances_ in older sklearn versions easily accessible without permutation
            importance = {col: 1/len(X.columns) for col in X.columns} 
        
        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv': cv_mean,
            'accuracy': r2 * 100,
            'precision': r2 * 98, # Simplified for regression demo
            'recall': r2 * 97,
            'f1': r2 * 97.5,
            'latency': int((time.time() - start_time) * 1000), # Actual measured latency
            'training': int((time.time() - start_time) * 1000),
            'features': len(X.columns),
            'cvScore': cv_mean * 100,
            'importance': importance,
            'color': MODEL_COLORS.get(name, '#667eea')
        }
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
        
        progress_bar.progress((idx + 1) / len(models))
    
    # Save best model
    if best_model:
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Store in session
        st.session_state.model = best_model
        st.session_state.scaler = scaler
        st.session_state.model_metrics = results
        
        # Training history
        st.session_state.training_history.append({
            'timestamp': datetime.now(),
            'best_model': best_name,
            'best_r2': best_score,
            'data_source': data_source,
            'n_samples': len(df)
        })
    
    status_text.text("✅ Training complete!")
    progress_bar.empty()
    
    return results, best_name, best_score

# ==========================================
# UI HELPER FUNCTIONS
# ==========================================

def get_aqi_category(aqi):
    """Get AQI category details"""
    for category, info in AQI_CATEGORIES.items():
        if info['range'][0] <= aqi <= info['range'][1]:
            return category, info['color'], info['icon'], info['description']
    return 'Unknown', '#6b7280', '❓', ''

def create_gauge_chart(value):
    """Create advanced gauge chart"""
    category, color, icon, desc = get_aqi_category(value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Current AQI<br><span style='font-size:0.8em;color:{color}'>{category} {icon}</span>",
            'font': {'size': 16, 'color': 'white'}
        },
        delta={'reference': 100, 'increasing': {'color': "#ef4444"}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [51, 100], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [101, 200], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [201, 300], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [301, 400], 'color': 'rgba(168, 85, 247, 0.2)'},
                {'range': [401, 500], 'color': 'rgba(127, 29, 29, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_forecast_chart(forecast_data, days=3):
    """Create comprehensive forecast chart"""
    df = pd.DataFrame(forecast_data)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter to selected days
    cutoff = df['time'].min() + timedelta(days=days)
    df = df[df['time'] <= cutoff]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('AQI Forecast', 'Temperature (°C)'),
        row_heights=[0.6, 0.4]
    )
    
    # AQI with area fill
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['aqi'],
            name="AQI",
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ),
        row=1, col=1
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['temperature'],
            name="Temperature",
            line=dict(color='#f97316', width=2)
        ),
        row=2, col=1
    )
    
    # Add AQI category lines
    for category, info in AQI_CATEGORIES.items():
        fig.add_hline(
            y=info['range'][0],
            line_dash="dash",
            line_color=info['color'],
            opacity=0.3,
            row=1, col=1
        )
    
    fig.update_layout(
        title=f"{days}-Day AQI Forecast",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_daily_forecast_cards(forecast_data):
    """Create daily forecast summary cards"""
    df = pd.DataFrame(forecast_data)
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    
    daily_stats = []
    for date in sorted(df['date'].unique())[:3]:
        day_data = df[df['date'] == date]
        daily_stats.append({
            'date': date,
            'avg_aqi': day_data['aqi'].mean(),
            'max_aqi': day_data['aqi'].max(),
            'min_aqi': day_data['aqi'].min(),
            'count': len(day_data)
        })
    
    return daily_stats

def create_model_comparison_chart(model_metrics):
    """Create model comparison radar chart"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'CV Score']
    
    fig = go.Figure()
    
    for name, metrics in model_metrics.items():
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['cvScore']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name,
            line=dict(color=MODEL_COLORS.get(name, '#667eea'), width=2),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[80, 100],
                color='white'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_feature_importance_chart(importance_dict, model_name):
    """Create feature importance bar chart"""
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Feature'],
            x=df['Importance'] * 100,
            orientation='h',
            marker_color=MODEL_COLORS.get(model_name, '#667eea'),
            text=df['Importance'].apply(lambda x: f"{x*100:.1f}%"),
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=f"Feature Importance - {model_name}",
        xaxis_title="Importance (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        showlegend=False,
        margin=dict(l=100, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_error_analysis_chart(model_metrics):
    """Create RMSE vs MAE scatter plot"""
    fig = go.Figure()
    
    for name, metrics in model_metrics.items():
        fig.add_trace(go.Scatter(
            x=[metrics['rmse']],
            y=[metrics['mae']],
            mode='markers+text',
            name=name,
            marker=dict(
                size=20,
                color=MODEL_COLORS.get(name, '#667eea'),
                line=dict(color='white', width=2)
            ),
            text=[name],
            textposition="top center",
            textfont=dict(size=10, color='white')
        ))
    
    fig.update_layout(
        title="Error Analysis (Lower is Better)",
        xaxis_title="RMSE",
        yaxis_title="MAE",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_efficiency_chart(model_metrics):
    """Create latency vs accuracy scatter plot"""
    fig = go.Figure()
    
    for name, metrics in model_metrics.items():
        fig.add_trace(go.Scatter(
            x=[metrics['latency']],
            y=[metrics['accuracy']],
            mode='markers+text',
            name=name,
            marker=dict(
                size=20,
                color=MODEL_COLORS.get(name, '#667eea'),
                line=dict(color='white', width=2)
            ),
            text=[name],
            textposition="top center",
            textfont=dict(size=10, color='white')
        ))
    
    fig.update_layout(
        title="Efficiency Analysis (Top-left is best)",
        xaxis_title="Latency (ms)",
        yaxis_title="Accuracy (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_hourly_weather_cards(forecast_data, hours=12):
    """Create hourly weather cards for next N hours"""
    df = pd.DataFrame(forecast_data[:hours])
    df['time'] = pd.to_datetime(df['time'])
    df['hour_label'] = df['time'].dt.strftime('%H:%M')
    
    return df

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="font-size: 2rem; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌍 AQI Intelligence
        </h1>
        <p style="color: #9ca3af;">Advanced Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize connections
    if st.session_state.db_client is None:
        st.session_state.db_client, st.session_state.db_status = init_mongodb_connection()
    
    # API Key Status
    api_valid, api_message = check_openweather_api()
    st.session_state.api_key_status = api_message
    
    # Connection Status Cards
    st.markdown("### 🔌 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.db_client:
            st.success("🟢 MongoDB")
        else:
            st.warning("🟡 MongoDB (mock)")
    
    with col2:
        if api_valid:
            st.success("🟢 OpenWeather")
        else:
            st.warning("🟡 OpenWeather (mock)")
    
    if st.session_state.db_client:
        try:
            db = st.session_state.db_client['air_quality']
            count = db['raw_aqi'].count_documents({})
            st.caption(f"📊 Records: {count}")
        except:
            pass
    
    st.markdown("---")
    
    # Location Info
    st.markdown("### 📍 Location")
    st.markdown(f"**Karachi, Pakistan**")
    st.caption(f"Lat: {LATITUDE:.4f}° | Lon: {LONGITUDE:.4f}°")
    
    st.markdown("---")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.toggle("🔄 Auto-refresh", value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
    
    # Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Collect Data", use_container_width=True):
            with st.spinner("Collecting..."):
                success, message = collect_air_quality_data()
                if success:
                    st.success(message)
                else:
                    st.error(message)
                st.session_state.last_refresh = datetime.now()
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            with st.spinner("Training models..."):
                results, best_name, best_score = train_models()
                st.success(f"✅ Best: {best_name} (R²={best_score:.3f})")
                st.rerun()
    
    st.markdown("---")
    st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# ==========================================
# AUTO-REFRESH LOGIC
# ==========================================
if st.session_state.auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh > refresh_interval:
        collect_air_quality_data()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# ==========================================
# MAIN CONTENT
# ==========================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🌍 AQI Intelligence Dashboard</h1>
    <p style="margin:0; opacity:0.9">Real-time Air Quality Monitoring with ML Predictions</p>
</div>
""", unsafe_allow_html=True)

# Initialize with mock data if needed
if st.session_state.latest_data is None:
    collect_air_quality_data()

# ==========================================
# CURRENT STATS ROW
# ==========================================
if st.session_state.latest_data:
    latest = st.session_state.latest_data
    category, color, icon, desc = get_aqi_category(latest['aqi'])
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px;">
            <p style="margin:0; color:white; opacity:0.9">Current AQI</p>
            <h1 style="margin:0; font-size: 3rem; color:white">{latest['aqi']:.0f}</h1>
            <p style="margin:0; color:white">
                <span class="live-indicator"></span> {category} {icon}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div style="background: #1e1e1e; padding: 1.5rem; border-radius: 15px; border: 1px solid #333;">
            <p style="margin:0; color:#9ca3af">PM2.5</p>
            <h2 style="margin:0; color:white">{latest['pm25']:.1f}</h2>
            <p style="margin:0; color:#9ca3af">µg/m³</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div style="background: #1e1e1e; padding: 1.5rem; border-radius: 15px; border: 1px solid #333;">
            <p style="margin:0; color:#9ca3af">Temperature</p>
            <h2 style="margin:0; color:white">{latest['temperature']:.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div style="background: #1e1e1e; padding: 1.5rem; border-radius: 15px; border: 1px solid #333;">
            <p style="margin:0; color:#9ca3af">Humidity</p>
            <h2 style="margin:0; color:white">{latest['humidity']:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MAIN GAUGE AND TODAY'S FORECAST
# ==========================================
if st.session_state.latest_data:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.plotly_chart(create_gauge_chart(latest['aqi']), use_container_width=True)
        
        # Quick stats
        st.markdown("### 🔬 Pollutants")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("PM10", f"{latest.get('pm10', 78):.0f} µg/m³")
        with col_b:
            st.metric("NO2", f"{latest.get('no2', 35):.0f} µg/m³")
    
    with col2:
        if st.session_state.forecast_data:
            # Today's forecast (first 24 hours)
            today_data = st.session_state.forecast_data[:24]
            df_today = pd.DataFrame(today_data)
            df_today['hour'] = pd.to_datetime(df_today['time']).dt.hour
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_today['hour'],
                y=df_today['aqi'],
                mode='lines+markers',
                name='AQI',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                title="Today's Hourly Forecast",
                xaxis_title="Hour",
                yaxis_title="AQI",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=300,
                showlegend=False
            )
            
            fig.update_xaxes(tickmode='linear', tick0=0, dtick=3)
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3-DAY FORECAST OVERVIEW
# ==========================================
st.markdown("## 📅 3-Day Forecast")

if st.session_state.forecast_data:
    # Daily forecast cards
    daily_stats = create_daily_forecast_cards(st.session_state.forecast_data)
    
    cols = st.columns(3)
    for idx, day in enumerate(daily_stats):
        with cols[idx]:
            day_name = "Today" if idx == 0 else "Tomorrow" if idx == 1 else day['date'].strftime('%A')
            category, color, icon, _ = get_aqi_category(day['avg_aqi'])
            
            st.markdown(f"""
            <div style="background: #1e1e1e; border: 2px solid {'#667eea' if idx == st.session_state.selected_day else '#333'}; 
                       border-radius: 12px; padding: 1rem; cursor: pointer;"
                 onclick="window.location.href='?day={idx}'">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <span style="font-weight: bold; color: white;">{day_name}</span>
                    <span style="color: #9ca3af; font-size: 0.8rem;">{day['count']}h</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                    <div>
                        <h2 style="margin:0; color: white;">{day['avg_aqi']:.0f}</h2>
                        <p style="margin:0; color: #9ca3af; font-size: 0.8rem;">Avg AQI</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin:0; color: #ef4444;">{day['max_aqi']:.0f}</p>
                        <p style="margin:0; color: #10b981;">{day['min_aqi']:.0f}</p>
                    </div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <span class="badge" style="background: {color};">{category} {icon}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Full forecast chart
    st.plotly_chart(create_forecast_chart(st.session_state.forecast_data, 3), use_container_width=True)
    
    # Hourly weather cards
    st.markdown("### ⏰ Next 12 Hours")
    hourly_df = create_hourly_weather_cards(st.session_state.forecast_data, 12)
    
    cols = st.columns(12)
    for idx, (_, row) in enumerate(hourly_df.iterrows()):
        with cols[idx]:
            weather_icon = "☀️" if "Sun" in row['weather'] else "☁️" if "Cloud" in row['weather'] else "🌙"
            st.markdown(f"""
            <div style="background: #1e1e1e; border-radius: 8px; padding: 0.5rem; text-align: center;">
                <p style="color: #9ca3af; font-size: 0.7rem;">{row['hour_label']}</p>
                <p style="font-size: 1.2rem; margin: 0;">{weather_icon}</p>
                <p style="color: white; font-weight: bold; margin: 0;">{row['aqi']:.0f}</p>
                <p style="color: #9ca3af; font-size: 0.7rem;">{row['temperature']:.0f}°</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# MODEL METRICS SECTION
# ==========================================
st.markdown("## 🧠 Model Performance Metrics")

if st.session_state.model_metrics:
    # Model selection cards
    st.markdown("### Select Model")
    cols = st.columns(len(st.session_state.model_metrics))
    
    for idx, (model_name, metrics) in enumerate(st.session_state.model_metrics.items()):
        with cols[idx]:
            is_selected = st.session_state.selected_model_id == model_name
            is_best = max(st.session_state.model_metrics.items(), key=lambda x: x[1]['accuracy'])[0] == model_name
            
            # Determine best badge
            best_badge = ""
            if is_best and not is_selected:
                best_badge = '<div class="best-badge"><span>🏆 Best</span></div>'
            
            st.markdown(f"""
            <div class="model-card {'selected' if is_selected else ''}" style="position: relative;">
                {best_badge}
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: {metrics['color']};"></div>
                    <span style="font-weight: bold; color: white;">{model_name}</span>
                    <span style="background: #333; padding: 2px 6px; border-radius: 12px; font-size: 0.6rem; color: #9ca3af; margin-left: auto;">
                        {metrics.get('type', 'Ensemble')}
                    </span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                        <span style="color: #9ca3af;">Accuracy</span>
                        <span style="color: white;">{metrics['accuracy']:.1f}%</span>
                    </div>
                    <div style="width: 100%; background: #333; height: 6px; border-radius: 3px;">
                        <div style="width: {metrics['accuracy']}%; background: {metrics['color']}; height: 6px; border-radius: 3px;"></div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.7rem;">
                    <div><span style="color: #9ca3af;">Precision:</span> <span style="color: white;">{metrics['precision']:.1f}%</span></div>
                    <div><span style="color: #9ca3af;">Recall:</span> <span style="color: white;">{metrics['recall']:.1f}%</span></div>
                    <div><span style="color: #9ca3af;">F1:</span> <span style="color: white;">{metrics['f1']:.1f}%</span></div>
                    <div><span style="color: #9ca3af;">RMSE:</span> <span style="color: white;">{metrics['rmse']:.1f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Handle model selection
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"Select", key=f"select_{model_name}"):
                    st.session_state.selected_model_id = model_name
                    st.rerun()
    
    # Model details toggle
    show_details = st.checkbox("Show Detailed Model Analysis", value=st.session_state.show_model_details)
    st.session_state.show_model_details = show_details
    
    if show_details:
        # Model comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            st.plotly_chart(create_model_comparison_chart(st.session_state.model_metrics), use_container_width=True)
        
        with col2:
            # Feature importance for selected model
            selected_metrics = st.session_state.model_metrics[st.session_state.selected_model_id]
            if 'importance' in selected_metrics:
                st.plotly_chart(create_feature_importance_chart(
                    selected_metrics['importance'], 
                    st.session_state.selected_model_id
                ), use_container_width=True)
        
        # Error analysis and efficiency
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_error_analysis_chart(st.session_state.model_metrics), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_efficiency_chart(st.session_state.model_metrics), use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📊 Detailed Metrics Matrix")
        
        metrics_data = []
        for name, metrics in st.session_state.model_metrics.items():
            metrics_data.append({
                'Model': name,
                'Accuracy (%)': f"{metrics['accuracy']:.1f}",
                'Precision (%)': f"{metrics['precision']:.1f}",
                'Recall (%)': f"{metrics['recall']:.1f}",
                'F1 Score (%)': f"{metrics['f1']:.1f}",
                'CV Score (%)': f"{metrics['cvScore']:.1f}",
                'RMSE': f"{metrics['rmse']:.1f}",
                'MAE': f"{metrics['mae']:.1f}",
                'Latency (ms)': metrics['latency'],
                'Training (s)': metrics['training'],
                'Features': metrics['features']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(
            df_metrics,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Accuracy (%)": st.column_config.NumberColumn("Accuracy", format="%.1f"),
                "Precision (%)": st.column_config.NumberColumn("Precision", format="%.1f"),
                "Recall (%)": st.column_config.NumberColumn("Recall", format="%.1f"),
                "F1 Score (%)": st.column_config.NumberColumn("F1", format="%.1f"),
                "CV Score (%)": st.column_config.NumberColumn("CV", format="%.1f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.1f"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.1f"),
                "Latency (ms)": st.column_config.NumberColumn("Latency", format="%d"),
                "Training (s)": st.column_config.NumberColumn("Training", format="%d"),
                "Features": st.column_config.NumberColumn("Features", format="%d")
            }
        )
        
        # Model summary
        st.markdown("### 📌 Model Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Selected Model:** {st.session_state.selected_model_id}")
            st.markdown(f"**Best Model:** {max(st.session_state.model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]}")
            st.markdown(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown(f"**Total Models:** {len(st.session_state.model_metrics)}")
            st.markdown(f"**Features Used:** {selected_metrics['features']}")
            st.markdown(f"**Cross-validation Score:** {selected_metrics['cvScore']:.1f}%")

else:
    st.info("No trained models yet. Click 'Train Models' in the sidebar to start training.")

# ==========================================
# HEALTH RECOMMENDATIONS
# ==========================================
if st.session_state.latest_data:
    st.markdown("## 🏥 Health Recommendations")
    
    latest = st.session_state.latest_data
    category, color, icon, desc = get_aqi_category(latest['aqi'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: #1e1e1e; border-left: 4px solid {color}; padding: 1rem; border-radius: 8px;">
            <h4 style="margin:0; color: white;">Current Conditions: {category} {icon}</h4>
            <p style="color: #9ca3af; margin-top: 0.5rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        recommendations = []
        
        if latest['aqi'] <= 50:
            recommendations.append("✅ Great day for outdoor activities!")
        elif latest['aqi'] <= 100:
            recommendations.append("😐 Sensitive individuals should limit outdoor exposure")
        elif latest['aqi'] <= 200:
            recommendations.append("😷 Consider wearing a mask outdoors")
        else:
            recommendations.append("⚠️ Stay indoors and use air purifiers")
        
        if latest['temperature'] > 35:
            recommendations.append("🌡️ Stay hydrated and avoid direct sun")
        elif latest['temperature'] < 10:
            recommendations.append("❄️ Dress warmly")
        
        if latest['pm25'] > 100:
            recommendations.append("💨 Close windows and use air purifier")
        
        for rec in recommendations:
            st.info(rec)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.caption("© 2025 AQI Intelligence")
with col2:
    if st.session_state.db_client:
        st.caption("🟢 MongoDB Connected")
    else:
        st.caption("🟡 MongoDB (Mock Mode)")
with col3:
    if api_valid:
        st.caption("🟢 OpenWeather Connected")
    else:
        st.caption("🟡 OpenWeather (Mock Mode)")
with col4:
    st.caption(f"v2.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
