import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def get_stock_data(ticker: str, start_date: str, end_date: str):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    data['Daily_Returns'] = data['Close'].pct_change()
    data['SMA_ratio'] = data['SMA_50'] / data['SMA_200']
    data['Volatility'] = data['Daily_Returns'].rolling(window=20).std()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Next_Day_Close'] = data['Close'].shift(-1)
    data['Target'] = (data['Next_Day_Close'] > data['Close']).astype(int)

    data = data.dropna()
    return data

def create_neural_network_model(input_dim: int):
    model = Sequential([
        Dense(units=16, activation='relu', input_dim=input_dim),
        Dense(units=8, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='BinaryCrossentropy',
        metrics=['accuracy'])

    return model              
        

def split_data_train_models(data: pd.DataFrame, test_size: float = 0.2):
    feature_cols = ['Daily_Returns', 'SMA_ratio', 'Volatility', 'RSI']
    X = data[feature_cols]
    y_price = data['Next_Day_Close']  
    y_signal = data['Target']  

    X_train, X_test, y_p_train, y_p_test, y_s_train, y_s_test = train_test_split(X, y_price, y_signal,test_size=test_size, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_p_train)
    price_predictions = lin_reg.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_p_test, price_predictions))

    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_s_train)
    signal_predictions = log_reg.predict(X_test_scaled)
    log_accuracy = accuracy_score(y_s_test, signal_predictions)

    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train_scaled, y_s_train)
    rf_predictions = rf_clf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_s_test, rf_predictions)
    feature_importances = dict(zip(feature_cols, rf_clf.feature_importances_))

    nn_model = create_neural_network_model(input_dim=X_train_scaled.shape[1])
    nn_model.fit(X_train_scaled, y_s_train, epochs=50, batch_size=16, verbose=1)
    nn_probs = nn_model.predict(X_test_scaled)
    nn_predictions = (nn_probs > 0.5).astype(int).flatten()
    nn_accuracy = accuracy_score(y_s_test, nn_predictions)

    latest_features_scaled = scaler.transform(data[feature_cols].iloc[-1:])
    latest_price_prediction = lin_reg.predict(latest_features_scaled)[0]
    latest_rf_signal = "Bullish(UP)" if rf_clf.predict(latest_features_scaled.reshape(1, -1))[0] == 1 else "Bearish(DOWN)"
    latest_nn_prob = float(nn_model.predict(latest_features_scaled)[0][0])

    metrics_summary = {
        "rmse_lin_reg": round(rmse, 2),
        "log_accuracy": round(log_accuracy*100, 2),
        "rf_accuracy": round(rf_accuracy*100, 2),
        "nn_accuracy": round(nn_accuracy*100, 2),
        "latest_price_prediction": round(latest_price_prediction, 2),
        "latest_rf_signal": latest_rf_signal,
        "latest_nn_prob": round(latest_nn_prob, 2),
        "latest_close_price": round(data['Close'].iloc[-1], 2),
        "latest_nn_confidence": round(latest_nn_prob*100, 2),
        "feature_importances": feature_importances
    }
    return metrics_summary  

def generate_stockmap_report(ticker: str, start_date: str, end_date: str, summary: dict) -> str:
    """Formats findings into a clean Markdown audit report."""
    report = f"""# 📈 StockMap AI — Quantitative Analysis & Trend Report

**Ticker Symbol:** `{ticker}`  
**Analysis Period:** {start_date} to {end_date}  
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

---

## Executive Market Summary

* **Latest Closing Price:** `${summary['latest_close_price']}`
* **Predicted Next-Day Target Price (Linear Regression):** `${summary['latest_price_prediction']}`
* **Random Forest Signal:** **{summary['latest_rf_signal']}**
* **TensorFlow Neural Network Bullish Probability:** `{summary['latest_nn_confidence']}%`

---

## Model Benchmark & Accuracy Summary

| Model Family | Algorithm Used | Evaluation Metric | Performance Score |
| :--- | :--- | :--- | :--- |
| **Price Target Engine** | Linear Regression | Root Mean Squared Error (RMSE) | **${summary['rmse_lin_reg']}** |
| **Linear Classifier** | Logistic Regression | Directional Accuracy | **{summary['log_accuracy']}%** |
| **Ensemble Trees** | Random Forest Classifier | Directional Accuracy | **{summary['rf_accuracy']}%** |
| **Deep Learning** | TensorFlow / Keras Sequential | Binary Cross-Entropy Accuracy | **{summary['nn_accuracy']}%** |

---

## Feature Importance (Random Forest Attribution)

Key technical drivers influencing stock direction:

"""
    for feat, weight in summary['feature_importances'].items():
        report += f"* **{feat}:** `{round(weight * 100, 2)}%` impact\n"

    report += f"""
---

## Technical Audit & Methodology Notes

1. **Gradient Descent & Standardization:** All continuous features (`SMA_Ratio`, `Volatility`, `RSI`) were scaled using Z-score standardization (`StandardScaler`) prior to training Logistic Regression and TensorFlow Neural Networks.
2. **Cost Function Optimization:** The TensorFlow network uses `binary_crossentropy` loss ($J(w,b)$) optimized via Adam optimizer over 50 epochs.
3. **Data Source:** Raw historical OHLCV data extracted directly from Yahoo Finance API (`yfinance`).

---
*Report generated automatically by StockMap AI Core (TensorFlow Engine).*
"""
    return report


ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")   
end_date = input("Enter the end date (YYYY-MM-DD): ")

stock_df = get_stock_data(ticker, start_date, end_date)
summary_metrics = split_data_train_models(stock_df)
final_report = generate_stockmap_report(ticker, start_date, end_date, summary_metrics)

print(final_report)
filename = f"{ticker}_StockMap_Report.md"
with open(filename, "w", encoding="utf-8") as f:
    f.write(final_report)
print(f"\nReport successfully saved to {filename}!")

    
    











