import requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, time_steps=60, target_indices=None):
    """
    Creates input sequences and target outputs for LSTM.

    Parameters:
    - data (np.array): Scaled data array.
    - time_steps (int): Number of past time steps to include.
    - target_indices (list): Indices of target parameters.

    Returns:
    - X (np.array): Input sequences.
    - y (np.array): Target outputs.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps][target_indices])
    return np.array(X), np.array(y)

def fetch_parameter_data(param, city, start_date, end_date, headers):
    base_url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "parameter": param,
        "date_from": start_date,
        "date_to": end_date,
        "limit": 10000,
        "page": 1
    }
    all_data = []
    while True:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            for result in results:
                if 'date' in result and 'utc' in result['date']:
                    all_data.append({
                        'datetime': result['date']['utc'],
                        'parameter': result['parameter'],
                        'value': result['value']
                    })
            params['page'] += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {param}: {e}")
            break
    return all_data

def fetch_openaq_data_parallel(city, parameters, start_date, end_date, api_key):
    import concurrent.futures
    headers = {"X-API-Key": api_key}
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_parameter_data, param, city, start_date, end_date, headers) for param in parameters]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            all_data.extend(data)
    if all_data:
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.pivot_table(values='value', index='datetime', columns='parameter', aggfunc='mean')
        df = df.resample('H').mean()
        return df
    else:
        return None

def build_multi_output_model(input_shape, num_outputs):
    """
    Builds and compiles a multi-output LSTM model.

    Parameters:
    - input_shape (tuple): Shape of the input data (time_steps, features).
    - num_outputs (int): Number of output parameters to predict.

    Returns:
    - model (Sequential): Compiled multi-output LSTM model.
    """
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_outputs))  # Output layer for multiple regression targets
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
