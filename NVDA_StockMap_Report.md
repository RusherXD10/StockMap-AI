# 📈 StockMap AI — Quantitative Analysis & Trend Report

**Ticker Symbol:** `NVDA`  
**Analysis Period:** 2023-01-01 to 2026-04-01  
**Report Generated:** 2026-10-09 23:02:39  

---

## Executive Market Summary

* **Latest Closing Price:** `$164.79`
* **Predicted Next-Day Target Price (Linear Regression):** `$130.74`
* **Random Forest Signal:** **Bullish(UP)**
* **TensorFlow Neural Network Bullish Probability:** `98.62%`

---

## Model Benchmark & Accuracy Summary

| Model Family | Algorithm Used | Evaluation Metric | Performance Score |
| :--- | :--- | :--- | :--- |
| **Price Target Engine** | Linear Regression | Root Mean Squared Error (RMSE) | **$63.68** |
| **Linear Classifier** | Logistic Regression | Directional Accuracy | **47.97%** |
| **Ensemble Trees** | Random Forest Classifier | Directional Accuracy | **49.59%** |
| **Deep Learning** | TensorFlow / Keras Sequential | Binary Cross-Entropy Accuracy | **49.59%** |

---

## Feature Importance (Random Forest Attribution)

Key technical drivers influencing stock direction:

* **Daily_Returns:** `26.33%` impact
* **SMA_ratio:** `24.07%` impact
* **Volatility:** `24.23%` impact
* **RSI:** `25.37%` impact

---

## Technical Audit & Methodology Notes

1. **Gradient Descent & Standardization:** All continuous features (`SMA_Ratio`, `Volatility`, `RSI`) were scaled using Z-score standardization (`StandardScaler`) prior to training Logistic Regression and TensorFlow Neural Networks.
2. **Cost Function Optimization:** The TensorFlow network uses `binary_crossentropy` loss ($J(w,b)$) optimized via Adam optimizer over 50 epochs.
3. **Data Source:** Raw historical OHLCV data extracted directly from Yahoo Finance API (`yfinance`).

---
*Report generated automatically by StockMap AI Core (TensorFlow Engine).*
