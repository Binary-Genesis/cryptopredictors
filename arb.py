import json
import time
from datetime import datetime
import requests
import pytz
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import schedule
json_file_path = 'arb.json'
base_url = 'https://api.coingecko.com/api/v3'
api_key = 'API_KEY'
json_file_path = 'arbitrum_cache.json'
def get_arbitrum_price():
    url = f'{base_url}/simple/price?ids=arbitrum&vs_currencies=usd'
    headers = {
        'Accepts': 'application/json',
        'X-CoinGecko-API-Key': api_key
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('arbitrum', {}).get('usd')
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def fetch_arbitrum_historical_data():
    timestamps, prices = [], []

    for _ in range(10):
        current_price = get_arbitrum_price()
        if current_price is not None:
            current_timestamp_utc = datetime.now(pytz.utc)
            current_timestamp_pst = current_timestamp_utc.astimezone(pytz.timezone('America/Los_Angeles'))
            timestamps.append(current_timestamp_pst.timestamp())
            prices.append(current_price)
            time.sleep(10)

    return timestamps, prices

def generate_features(timestamps):
    return np.array(timestamps).reshape(-1, 1)

def train_arbitrum_model(X, y):
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    model.fit(X, y)
    return model

def train_arbitrum_random_forest_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_arbitrum_gradient_boosting_model(X, y):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    return model

def make_arbitrum_predictions(models):
    timestamps, _ = fetch_arbitrum_historical_data()

    if timestamps and len(timestamps) > 0:
        X = generate_features(np.array(timestamps).reshape(-1, 1))

        predictions_data = {}

        for model_name, model in models.items():
            predictions = model.predict(X)
            # Convert NumPy array to Python list before saving to JSON
            predictions_data[model_name] = predictions.tolist()

        # Save predictions to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(predictions_data, json_file)

def main():
    models = {}
    timestamps, prices = fetch_arbitrum_historical_data()

    if timestamps and len(timestamps) > 0:
        X = generate_features(np.array(timestamps).reshape(-1, 1))
        y = np.array(prices)

        models['Ridge'] = train_arbitrum_model(X, y)
        models['RandomForest'] = train_arbitrum_random_forest_model(X, y)
        models['GradientBoosting'] = train_arbitrum_gradient_boosting_model(X, y)

    # Schedule predictions every hour
    schedule.every().hour.at(":00").do(make_arbitrum_predictions, models=models)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
