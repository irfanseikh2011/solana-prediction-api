import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from collections import deque
import time
import os
import xgboost as xgb
import warnings
import websocket # pip install websocket-client
import json
from datetime import datetime
from app import latest_prediction_data, data_lock, start_api
import threading
import queue # For inter-thread communication

warnings.filterwarnings('ignore') # Suppress warnings

# --- Configuration ---
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
SYMBOL = 'SOLUSDT'
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK_HISTORY_DAYS_INITIAL_TRAIN = 30 # For initial training
LOOKBACK_PERIOD_FOR_PREDICTION = 120 # Number of minutes needed for features
RETRAIN_AFTER_N_PREDICTIONS = 10
RECENT_OUTCOMES_BUFFER_SIZE = 1000 # Number of past outcomes for retraining
PROFIT_THRESHOLD_PC = 0.0005 # 0.05% for a meaningful LONG/SHORT

# Initialize Binance Client for REST API calls (historical, evaluation)
client = Client(API_KEY, API_SECRET)

# Global deque to hold the incoming OHLCV data from WebSocket
# This will replace the need for get_latest_klines for feature calculation
ohlcv_buffer_deque = deque(maxlen=LOOKBACK_PERIOD_FOR_PREDICTION)
# Queue for closed candles to be processed by the main thread
closed_candle_queue = queue.Queue()

# --- 1. Data Acquisition (REST API for historical and evaluation) ---
def get_historical_klines(symbol, interval, lookback_days):
    print(f"Fetching {interval} historical klines for {symbol} for last {lookback_days} days...")
    end_str = "now"
    start_str = f"{lookback_days} days ago UTC"
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df.set_index('Open Time', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def get_specific_kline(symbol, interval, start_time):
    """Fetches a specific kline that starts at or after start_time."""
    
    # Convert pandas datetime to milliseconds for Binance API
    start_time_ms = int(start_time.timestamp() * 1000)
    
    klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_time_ms, limit=2) # Get 2 for safety
    
    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    if df.empty:
        return None
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    # Filter for the exact kline we need (matching start_time)
    matching_kline = df[df['Open Time'] == start_time]
    if not matching_kline.empty:
        return matching_kline.iloc[0]
    return None


# --- WebSocket Callbacks ---
def on_message(ws, message):
    json_message = json.loads(message)
    # Check if it's a kline event
    if 'k' in json_message:
        candle = json_message['k']
        if candle['x']: # 'x' indicates if the candle is closed
            # Extract OHLCV data for the closed candle
            new_candle = {
                'Open Time': pd.to_datetime(candle['t'], unit='ms'),
                'Open': float(candle['o']),
                'High': float(candle['h']),
                'Low': float(candle['l']),
                'Close': float(candle['c']),
                'Volume': float(candle['v'])
            }
            closed_candle_queue.put(new_candle) # Put into queue for main thread processing

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket Closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket Opened.")

# --- 2. Feature Engineering ---
def add_features(df):
    """Adds technical and supply/demand proxy features to the DataFrame."""

    # Ensure index is sorted for rolling calculations
    df = df.sort_index()

    # Basic Price Action
    df['Return_1m'] = df['Close'].pct_change()
    df['Range_1m'] = (df['High'] - df['Low']) / df['Close']
    df['Body_1m'] = abs(df['Close'] - df['Open']) / df['Close']
    # Add small epsilon to avoid division by zero for wick ratios
    df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-9)
    df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-9)

    # Volume Features
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Relative'] = df['Volume'] / df['Volume_MA_20']

    # Momentum Features (price change over X minutes)
    for p in [5, 10, 15, 30]:
        df[f'Price_Change_{p}m'] = df['Close'].pct_change(periods=p)
        df[f'Volatility_{p}m_std'] = df['Close'].rolling(window=p).std() / df['Close'].shift(p)

    # Technical Indicators (TA library)
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    for p in [10, 20, 50, 100]:
        df[f'EMA_{p}'] = ta.trend.EMAIndicator(df['Close'], window=p).ema_indicator()
        df[f'Close_vs_EMA_{p}'] = (df['Close'] - df[f'EMA_{p}']) / df[f'EMA_{p}']

    # Rolling VWAP as a S&D proxy (needs Volume column)
    for p in [20, 50, 100]:
        price_volume = df['Close'] * df['Volume']
        df[f'VWAP_rolling_{p}'] = price_volume.rolling(window=p).sum() / (df['Volume'].rolling(window=p).sum() + 1e-9) # Add epsilon
        df[f'Close_vs_VWAP_{p}'] = (df['Close'] - df[f'VWAP_rolling_{p}']) / (df[f'VWAP_rolling_{p}'] + 1e-9) # Add epsilon

    # Price action within recent window (simplified S&D proxies)
    for window in [5, 10, 20]:
        df[f'High_vs_MA_High_{window}'] = (df['High'] - df['High'].rolling(window=window).mean()) / (df['High'].rolling(window=window).mean() + 1e-9)
        df[f'Low_vs_MA_Low_{window}'] = (df['Low'] - df['Low'].rolling(window=window).mean()) / (df['Low'].rolling(window=window).mean() + 1e-9)
        df[f'Close_vs_Highest_{window}m'] = (df['Close'] - df['High'].rolling(window=window).max()) / (df['High'].rolling(window=window).max() + 1e-9)
        df[f'Close_vs_Lowest_{window}m'] = (df['Close'] - df['Low'].rolling(window=window).min()) / (df['Low'].rolling(window=window).min() + 1e-9)
        # Binary features for recent high/low
        df[f'Is_Near_High_{window}m'] = (df['Close'] >= df['High'].rolling(window=window).max() * 0.999).astype(int)
        df[f'Is_Near_Low_{window}m'] = (df['Close'] <= df['Low'].rolling(window=window).min() * 1.001).astype(int)


    return df

# --- 3. Target Variable Creation ---
def create_target(df):
    """Creates the target variable for the next 5 minutes."""
    df['future_close_5m'] = df['Close'].shift(-5)
    df['price_change_5m'] = (df['future_close_5m'] - df['Close']) / df['Close']

    df['target'] = np.nan
    df.loc[df['price_change_5m'] > PROFIT_THRESHOLD_PC, 'target'] = 1  # LONG
    df.loc[df['price_change_5m'] < -PROFIT_THRESHOLD_PC, 'target'] = 0 # SHORT

    # Drop rows where target is NaN (neutral movements)
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int) # Convert to int

    # Drop future-looking columns before returning
    df.drop(columns=['future_close_5m', 'price_change_5m'], inplace=True, errors='ignore')
    return df

# --- 4. Model Training and Prediction ---
class TradingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.fitted_scaler = False # Flag to check if scaler has been fitted
        self.feature_columns = None
        self.recent_outcomes_buffer = deque(maxlen=RECENT_OUTCOMES_BUFFER_SIZE)
        self.predictions_made = 0
        self.correct_predictions = 0

    def prepare_data(self, df):
        """Prepare features and target for training/prediction."""
        df_processed = add_features(df.copy())
        
        # Determine feature columns from initial training
        if self.feature_columns is None:
            # All columns except Open, High, Low, Close, Volume, and target
            non_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target'] # 'target' might not be present initially
            self.feature_columns = [col for col in df_processed.columns if col not in non_feature_cols]
            print(f"Identified {len(self.feature_columns)} features.")

        # Ensure all feature columns exist, fill missing (e.g., from small lookback) with 0 or mean
        # This is crucial if a specific feature cannot be calculated due to insufficient lookback
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0 # Fill with 0 for safety, but data processing should prevent this
            else:
                df_processed[col] = df_processed[col].fillna(0) # Fill NaNs within identified features

        # Drop rows with NaN if they occur after feature engineering due to lookback that couldn't be filled
        # We also need to drop NaNs in the target column if present
        # This step is important if features have NaNs at the beginning of the dataframe
        df_processed.dropna(subset=self.feature_columns, inplace=True)
        
        return df_processed

    def train_model(self, df_train):
        """Trains or re-trains the XGBoost model."""
        print("Training model...")
        
        if 'target' not in df_train.columns:
            print("Error: train_model received DataFrame without 'target' column.")
            return

        X = df_train[self.feature_columns]
        y = df_train['target']

        if len(X) == 0:
            print("No valid feature rows after processing for training. Cannot train model.")
            self.model = None # Reset model if no data
            return

        # Scale features
        # CORRECTED: Use self.fitted_scaler flag consistently
        if not self.fitted_scaler: 
            self.scaler.fit(X)
            self.fitted_scaler = True 
        X_scaled = self.scaler.transform(X)

        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                                       use_label_encoder=False, 
                                       n_estimators=100, learning_rate=0.1, max_depth=5,
                                       random_state=42, tree_method='hist') # Use hist for faster training on large data
        self.model.fit(X_scaled, y)
        print(f"Model trained on {len(X)} samples.")
        
        # Simple evaluation on training data
        train_preds = self.model.predict(X_scaled)
        train_accuracy = np.mean(train_preds == y)
        print(f"Training accuracy: {train_accuracy:.2f}")


    def initial_train(self):
        """Performs initial training on a larger historical dataset."""
        df_hist = get_historical_klines(SYMBOL, INTERVAL, LOOKBACK_HISTORY_DAYS_INITIAL_TRAIN)
        df_hist_processed = self.prepare_data(df_hist) # This populates self.feature_columns
        df_labeled = create_target(df_hist_processed.copy())

        # Populate the recent outcomes buffer for first retraining cycles
        if self.feature_columns: # Ensure features are identified
            for index, row in df_labeled.tail(RECENT_OUTCOMES_BUFFER_SIZE).iterrows():
                # Only add rows where all feature columns are valid and not NaN
                if not any(pd.isna(row[self.feature_columns])):
                    self.recent_outcomes_buffer.append((row[self.feature_columns].values, int(row['target'])))
                else:
                    print(f"Warning: Skipping row with NaN features in initial buffer population at {index}.")

        self.train_model(df_labeled)

    def predict(self, df_current_klines_buffer):
        """
        Makes a prediction for the latest candle in the provided DataFrame buffer.
        Returns prediction (int), label (str), and the features array used for prediction (numpy array).
        """
        if self.model is None:
            print("Model not trained yet. Cannot predict.")
            return None, None, None

        # df_current_klines_buffer is already a DataFrame from the deque
        df_features = self.prepare_data(df_current_klines_buffer.copy())
        
        # We need at least LOOKBACK_PERIOD_FOR_PREDICTION to compute all features reliably
        if df_features.empty or len(df_features) < LOOKBACK_PERIOD_FOR_PREDICTION: # Ensure enough historical context
            print("Not enough valid data after feature engineering for prediction.")
            return None, None, None
            
        # The features for the latest closed candle are the last row
        latest_features_series = df_features[self.feature_columns].iloc[-1]
        latest_features_numpy = latest_features_series.values.reshape(1, -1) # For scaling and prediction

        if not self.fitted_scaler:
            print("Scaler has not been fitted. Skipping prediction.")
            return None, None, None

        latest_features_scaled = self.scaler.transform(latest_features_numpy)

        prediction = self.model.predict(latest_features_scaled)[0]
        prediction_proba = self.model.predict_proba(latest_features_scaled)[0]

        direction = "LONG" if prediction == 1 else "SHORT"
        print(f"Prediction: {direction} (Confidence: {prediction_proba.max():.2f})")
        
        # Return the raw numpy array of features used for this prediction
        return prediction, direction, latest_features_series.values 

    def evaluate_prediction(self, predicted_time, predicted_direction_label, 
                            initial_pred_price, predicted_dir_int, 
                            features_at_prediction_time): # This argument now holds the features array
        """Evaluates a past prediction."""
        
        # Calculate the target kline's expected start time
        # The prediction is for the close price *after* the current candle closes (which is already done)
        # So, the 5m target is the candle starting at predicted_time + 5 minutes
        target_kline_start_time = predicted_time + pd.Timedelta(minutes=5)
        
        # Wait until the target kline should have closed
        wait_for_secs = (target_kline_start_time + pd.Timedelta(minutes=1, seconds=10)).timestamp() - time.time()
        if wait_for_secs > 0:
            print(f"Waiting {int(wait_for_secs)} seconds for 5-minute outcome kline {target_kline_start_time.strftime('%H:%M')} to close...")
            time.sleep(wait_for_secs)

        # Fetch the specific target kline using REST API
        target_kline = get_specific_kline(SYMBOL, INTERVAL, target_kline_start_time)

        if target_kline is None:
            print(f"Error: Could not retrieve 5-minute outcome kline for {target_kline_start_time}. Skipping evaluation and buffer update.")
            return False # Indicate failure

        actual_close_price = target_kline['Close']
        actual_price_change = (actual_close_price - initial_pred_price) / initial_pred_price

        # Determine actual outcome based on profit threshold
        actual_outcome_int = np.nan
        if actual_price_change > PROFIT_THRESHOLD_PC:
            actual_outcome_int = 1 # LONG
        elif actual_price_change < -PROFIT_THRESHOLD_PC:
            actual_outcome_int = 0 # SHORT
        else:
            actual_outcome_int = 1 if actual_price_change >= 0 else 0 # Simplified binary, close to neutral counts as non-predicted direction

        is_correct = (predicted_dir_int == actual_outcome_int)
        
        # Add to buffer if a clear outcome that matches our target definition occurred
        if not np.isnan(actual_outcome_int):
            self.recent_outcomes_buffer.append((features_at_prediction_time, actual_outcome_int))
        else:
            print("Evaluation: Outcome was too neutral, not added to retraining buffer.")
            
        self.predictions_made += 1
        current_accuracy = (self.correct_predictions / self.predictions_made) * 100
        if is_correct:
            self.correct_predictions += 1
            print(f"  Outcome: Correct! (Predicted: {predicted_direction_label}, Actual: {'LONG' if actual_outcome_int == 1 else 'SHORT'}, Change: {actual_price_change:.4%})")
        else:
            print(f"  Outcome: Incorrect! (Predicted: {predicted_direction_label}, Actual: {'LONG' if actual_outcome_int == 1 else 'SHORT'}, Change: {actual_price_change:.4%})")
            
        print(f"Total Predictions: {self.predictions_made}, Correct: {self.correct_predictions}, Accuracy: {current_accuracy:.2f}%")

        return is_correct

# --- Main Execution Loop ---
def run_predictor():
    predictor = TradingPredictor()
    print("Performing initial training...")
    
    predictor.initial_train() # This now handles both, including buffer loading

    # After initial training, populate the live OHLCV buffer with recent klines
    recent_historical_klines = get_historical_klines(SYMBOL, INTERVAL, 
                                                     # Need enough minutes for the lookback period
                                                     LOOKBACK_PERIOD_FOR_PREDICTION // 60 + 1 
                                                    )
    
    # Add each row to the deque, ensuring it serves as a base for feature calculation
    # Only append the actual OHLCV columns
    for idx, row in recent_historical_klines.tail(LOOKBACK_PERIOD_FOR_PREDICTION).iterrows():
        ohlcv_buffer_deque.append({
            'Open Time': idx, 'Open': row['Open'], 'High': row['High'], 
            'Low': row['Low'], 'Close': row['Close'], 'Volume': row['Volume']
        })
    print(f"OHLCV buffer initialized with {len(ohlcv_buffer_deque)} historical candles.")


    # 2. Start WebSocket in a separate thread
    socket_url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{INTERVAL}"
    ws = websocket.WebSocketApp(socket_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # Start the WebSocket in a daemon thread so it exits when main thread exits
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    print(f"WebSocket started for {SYMBOL} {INTERVAL} at {socket_url}")

    prediction_counter = 0

    while True:
        try:
            # 3. Wait for a new closed candle from the WebSocket via the queue
            print("Waiting for a new closed candle...")
            new_candle_data = closed_candle_queue.get(timeout=70) # Wait slightly longer than 1 minute

            # Add the new candle to the deque
            ohlcv_buffer_deque.append(new_candle_data)
            
            # Construct DataFrame from the deque for feature calculation
            current_klines_df = pd.DataFrame(list(ohlcv_buffer_deque))
            current_klines_df.set_index('Open Time', inplace=True)
            current_klines_df.sort_index(inplace=True) # Ensure sorted by time
            
            # The latest candle in the DF is the one we just received and will predict on
            current_candle_start_time = current_klines_df.index[-1]
            current_close_price = current_klines_df['Close'].iloc[-1]

            print(f"\n--- Processing new candle at {current_candle_start_time} (Close: {current_close_price}) ---")
            
            # 4. Make prediction using the updated buffer DataFrame
            predicted_dir_int, predicted_dir_label, features_used_for_prediction = predictor.predict(current_klines_df)

            with data_lock:
                latest_prediction_data["price"] = current_close_price
                latest_prediction_data["signal"] = predicted_dir_label
                latest_prediction_data["confidence"] = float(100 if predicted_dir_int is not None else 0)  # You may replace this with actual confidence if you return it
                latest_prediction_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
                latest_prediction_data["supply_zones"] = [[175.2, 176.5]]  # Placeholder
                latest_prediction_data["demand_zones"] = [[169.0, 170.3]]


            # Ensure we have enough data points in the buffer for feature calculation
            if len(current_klines_df) < LOOKBACK_PERIOD_FOR_PREDICTION:
                print(f"Not enough data in OHLCV buffer ({len(current_klines_df)}/{LOOKBACK_PERIOD_FOR_PREDICTION}) for reliable features. Skipping prediction.")
                continue # Skip prediction for this candle until buffer is full

            if predicted_dir_int is None or features_used_for_prediction is None:
                print("Skipping prediction for this candle due to data or model issues.")
                continue

            # 5. Evaluate this prediction after 5 minutes
            predictor.evaluate_prediction(current_candle_start_time, predicted_dir_label, 
                                          current_close_price, predicted_dir_int, 
                                          features_used_for_prediction) # Pass features directly for buffer update

            prediction_counter += 1

            # 6. Retrain the model periodically
            if prediction_counter % RETRAIN_AFTER_N_PREDICTIONS == 0:
                print(f"\n--- Retraining model after {prediction_counter} predictions ---")
                
                buffer_df_list = []
                for features_arr, target_val in predictor.recent_outcomes_buffer:
                    if predictor.feature_columns and len(features_arr) == len(predictor.feature_columns): # Sanity check
                        row_dict = dict(zip(predictor.feature_columns, features_arr))
                        row_dict['target'] = target_val
                        buffer_df_list.append(row_dict)
                    else:
                        print(f"Skipping malformed features array in buffer: expected {len(predictor.feature_columns)}, got {len(features_arr)}")
                
                if buffer_df_list:
                    retrain_df = pd.DataFrame(buffer_df_list)
                    predictor.train_model(retrain_df)
                else:
                    print("Recent outcomes buffer is empty, skipping retraining.")

        except queue.Empty:
            print("No new candle received from WebSocket within timeout. Checking connection status...")
            if not ws_thread.is_alive():
                print("WebSocket thread died. Attempting to restart...")
                ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
                ws_thread.start()
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to resume gracefully after 5 seconds...")
            time.sleep(5)



if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_api, daemon=True)
    flask_thread.start()

    run_predictor()
