import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime
import time
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


# Function to fetch data from Alpha Vantage API
def fetch_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

    # Fetch the data
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# Your API key here
api_key = '9ZOGJND3TJGY4O33'

# Fetch data for AAPL (Apple) from Alpha Vantage API
symbol = 'AAPL'
data = fetch_data(symbol, api_key)

# If data is fetched successfully, process it
if data:
    # Assuming that the data is under 'Time Series (Daily)' key
    time_series = data.get('Time Series (Daily)', {})

    # Convert time series data to a DataFrame
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for date, daily_data in time_series.items():
        df = df._append({
            'Date': date,
            'Open': float(daily_data['1. open']),
            'High': float(daily_data['2. high']),
            'Low': float(daily_data['3. low']),
            'Close': float(daily_data['4. close']),
            'Volume': int(daily_data['5. volume'])
        }, ignore_index=True)

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort data by date in ascending order
    df = df.sort_values(by='Date')

    # Display Top 5 Rows
    print("Top 5 Rows of the Dataset:")
    print(df.head())

    # Display Last 5 Rows
    print("\nLast 5 Rows of the Dataset:")
    print(df.tail())

    # Find Shape of the Dataset
    print("\nShape of the Dataset (Number of Rows and Number of Columns):")
    print("Number of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    # Train-test split
    training_data_len = int(np.ceil(len(scaled_data) * .95))
    train_data = scaled_data[0:training_data_len, :]

    # Prepare features and labels for training
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # LSTM Model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10)

    # Prepare test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = df['Close'][training_data_len:].values
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plotting the actual vs predicted stock prices
    plt.figure(figsize=(10, 8))
    plt.plot(df['Date'][:training_data_len], df['Close'][:training_data_len], color='blue', label='Training Data')
    plt.plot(df['Date'][training_data_len:], df['Close'][training_data_len:], color='red', label='Actual Stock Price')
    plt.plot(df['Date'][training_data_len:], predictions, color='green', label='Predicted Stock Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Add delay before making another request (to avoid hitting the API request limit)
    time.sleep(60)  # Sleep for 60 seconds between requests
else:
    print("Failed to fetch data.")
