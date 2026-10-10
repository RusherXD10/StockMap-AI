# 📈 StockMap AI — Quantitative Stock Analysis & Trend Prediction

StockMap AI is a Python-based stock market analysis project that combines technical indicators, machine learning, and deep learning to analyze historical stock prices and estimate future price movements.

The project uses historical market data from Yahoo Finance to engineer technical features, train multiple predictive models, evaluate their performance, and generate a structured Markdown report containing predictions, directional signals, model metrics, and feature importance.

> **Project status:** Learning and development project. Predictions are experimental and are not financial advice.

## 🎯 Project Objectives

- Retrieve historical stock market data using Yahoo Finance.
- Engineer technical indicators and statistical features from historical prices.
- Train and evaluate multiple machine-learning and deep-learning models.
- Estimate the next trading day's closing price.
- Classify the expected next-day price direction as bullish or bearish.
- Compare model performance using appropriate evaluation metrics.
- Automatically generate a Markdown report summarizing the analysis.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| yfinance | Download historical stock market data |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning, feature scaling, and evaluation |
| TensorFlow / Keras | Neural network development and training |
| Markdown | Automated analysis report generation |

## 📊 Technical Indicators & Feature Engineering

StockMap AI derives the following features from historical closing prices:

- **SMA 50:** 50-trading-day Simple Moving Average.
- **SMA 200:** 200-trading-day Simple Moving Average.
- **Daily Returns:** Percentage change in closing price between consecutive trading sessions.
- **SMA Ratio:** Ratio of the 50-day SMA to the 200-day SMA.
- **Volatility:** Rolling 20-trading-day standard deviation of daily returns.
- **RSI:** Relative Strength Index calculated using a 14-period rolling average of gains and losses.

The model uses Daily Returns, SMA Ratio, Volatility, and RSI as input features.

The target variables are:
- **Price target:** The next trading day's closing price.
- **Directional target:** A binary label indicating whether the next closing price is higher than the current closing price.

Rows containing missing values resulting from rolling calculations and the next-day target are removed before training.

## 🤖 Machine Learning Models

StockMap AI implements four predictive models.

### 1. Linear Regression
Predicts the next trading day's closing price using engineered technical features.

**Evaluation metric:** Root Mean Squared Error (RMSE).

### 2. Logistic Regression
Classifies the expected next-day price direction into two classes:
- `1` — Bullish: next closing price is higher.
- `0` — Bearish or non-increasing: next closing price is not higher.

**Evaluation metric:** Classification accuracy.

### 3. Random Forest Classifier
Uses an ensemble of decision trees to predict price direction. It also provides feature-importance scores to help identify which input features contribute most to the fitted model.

**Evaluation metrics:** Classification accuracy and feature importance.

### 4. Neural Network (TensorFlow / Keras)
A feed-forward neural network with two hidden layers:
- First hidden layer: 16 neurons with ReLU activation.
- Second hidden layer: 8 neurons with ReLU activation.
- Output layer: 1 neuron with sigmoid activation.

The network is trained using the Adam optimizer and binary cross-entropy loss over 50 epochs, with a batch size of 16.

**Evaluation metric:** Classification accuracy.

## 📁 Project Structure

```text
StockMap-AI/
├── data/                       # Stock data storage
├── reports/                    # Generated analysis reports
├── tests/                      # Unit tests and validation
├── stockmap_engine.py          # Main analysis and modeling pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files excluded from Git
```

The `data/`, `reports/`, and `tests/` directories are intended for their respective purposes. The current script saves its generated report in the working directory unless configured to save it under `reports/`.

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or another Python version compatible with your installed TensorFlow release.
- Git.
- An internet connection to retrieve historical stock data.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/StockMap-AI.git
cd StockMap-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the libraries required by the project, including `yfinance`, `pandas`, `numpy`, `scikit-learn`, and `tensorflow`.

## ▶️ Running the Project

Run the main Python script from the project root:

```bash
python stockmap_engine.py
```

The program prompts you to enter a stock ticker symbol and an analysis period.

**Example input:**

```text
Enter the stock ticker symbol (e.g., AAPL): NVDA
Enter the start date (YYYY-MM-DD): 2022-01-01
Enter the end date (YYYY-MM-DD): 2025-01-01
```

The script then:

1. Downloads historical stock data.
2. Calculates technical indicators and prepares the dataset.
3. Splits the data chronologically into training and testing sets.
4. Standardizes the input features using `StandardScaler`.
5. Trains the four predictive models.
6. Evaluates model performance on the test set.
7. Generates a Markdown report containing the latest price, model predictions, classification metrics, and Random Forest feature importance.

The generated report is named using the ticker symbol, for example:

`NVDA_StockMap_Report.md`

## 📑 Report Contents

Each generated report includes:

- Stock ticker and analysis period.
- Latest closing price.
- Estimated next-day closing price.
- Random Forest directional prediction.
- Neural network output probability.
- Linear Regression RMSE.
- Classification accuracy for Logistic Regression, Random Forest, and the neural network.
- Random Forest feature-importance scores.
- Methodology and data-source notes.

## 📈 Evaluation Methodology

The dataset is split chronologically using `shuffle=False`, preserving the order of historical observations.

The input features are standardized by fitting `StandardScaler` on the training data and applying the same transformation to the test data.

The project evaluates:
- **RMSE** for next-day closing-price prediction.
- **Classification accuracy** for directional predictions.
- **Feature importance** for interpreting the Random Forest model.

Model performance can vary depending on the ticker, analysis period, market conditions, and selected features. Evaluation results from historical data do not guarantee future predictive performance.

## 🔮 Future Improvements

Potential improvements to the project include:

- Backtesting trading strategies with transaction costs.
- Comparing additional technical indicators and feature sets.
- Tuning hyperparameters through time-series-aware validation.
- Evaluating classification models with precision, recall, F1-score, and ROC-AUC.
- Comparing predictions against simple baseline models.
- Improving neural network calibration and probability evaluation.
- Saving reports directly to the `reports/` directory.
- Adding automated tests and visualizations.
- Building an interactive dashboard for stock analysis.

## ⚠️ Disclaimer

StockMap AI is an educational and experimental machine-learning project. It is not a financial advisory service, and its predictions should not be used as the sole basis for investment decisions. Historical patterns may not persist, and all predictive models are subject to uncertainty.

## 👨‍💻 Author

Developed as a project to explore Python, quantitative analysis, feature engineering, machine learning, and deep learning in financial-market applications.

**Technologies:** Python · Pandas · NumPy · Scikit-learn · TensorFlow · Yahoo Finance

