# TradeAI System Report

## 1. Executive Summary

**TradeAI** is a modular, production-ready AI platform designed for Forex trading analysis and prediction. It aims to bridge the gap between complex quantitative algorithmic models and human-readable trading insights. The system integrates deep learning (specifically LSTM neural networks) for raw price prediction, technical indicator feature engineering, a comprehensive risk management engine, and a Large Language Model (LLM) layer that explains the model outputs in plain, understandable language.

The platform is designed to be highly modular, supporting multi-timeframe and multi-pair (e.g., EUR/USD, GBP/USD, USD/JPY) analysis. It is equipped with a modern web-based dashboard for traders to manually interact with the models or request real-time analyses.

---

## 2. Core Architecture & Components

The application architecture is logically separated into the following key layers:

### A. Frontend & Interface Layer
* **Django Templates & Dashboard**: The primary interface is a traditional Django server-side rendered application (`AgentApp`) paired with asynchronous JavaScript to communicate with backend APIs.
* **Trader Profiles**: Users have a comprehensive profile (`TraderProfile` model) that tracks performance metrics (win rate, Sharpe ratio, max drawdown) and stores personal configuration for risk management (e.g., auto-trading, fixed vs. risk-based lot sizes, maximum risk per trade).

### B. Backend API Layer
The Django backend (`AgentApp/views.py`) exposes several endpoints that form the backbone of the application's intelligence:
* `/api/chat/`: A legacy endpoint offering a chat-style prompt where the user can query the system along with a prediction request.
* `/api/predict/`: A structured API that processes a pair and timeframe (e.g., `AUD/USD 1h`), runs inference, manages risk, and directly returns a detailed JSON object containing signals, take profits, stop losses, and an LLM-generated string of reasoning for the trade.
* `/api/backtest/`: Executes a historical simulation of the AI models over a desired timeframe, generating metrics, an equity curve, and a trade log to evaluate the effectiveness of the model strategies.

### C. Data Ingestion & Preprocessing Layer
* **Twelve Data Integration (`loader.py`, `client.py`)**: The system fetches historical and real-time OHLCV (Open, High, Low, Close, Volume) data from the Twelve Data API. 
* **Feature Engineering (`processing.py`, `indicators.py`)**: Before data ever touches a model, it runs through an extensive feature engineering pipeline. The system calculates a variety of robust technical indicators:
  * **Momentum**: RSI, MACD
  * **Trend**: EMA crossovers (e.g., EMA 7, 20, 50)
  * **Volatility**: ATR (Average True Range), Bollinger Bands
  * **Regime Detection**: Detects market states (trending up, trending down, ranging)

### D. Machine Learning & Predictive Pipeline
* **Dynamic Model Registry (`registry.py`)**: Manages the loading and saving of different models specifically trained for corresponding currency pairs and timeframes.
* **Deep Learning Model (`modeltrain.py`)**: Uses a Sequential deep learning model implemented via `TensorFlow` and `Keras`. The architecture primarily uses **LSTM (Long Short-Term Memory)** layers designed for time-series and sequential data, followed by dropout layers for regularization, ending in Dense layers. 
* **Inference Pipeline (`engine.py`, `inference.py`)**: Responsible for bringing data and models together. Unseen data is padded, scaled (or left unscaled depending on configuration), and processed through the LSTM to retrieve a raw directional probability that translates into a `BUY`, `SELL`, or `HOLD` signal.

### E. Risk Management Engine
* **`risk_engine.py`**: A critical component bridging the model prediction and the actual trade execution. It takes basic signals and applies rigid risk management formulas based on the trader's profile.
* It calculates dynamic **Stop Loss (SL)** and **Take Profit (TP)** levels primarily utilizing current volatility metrics like the ATR (Average True Range).
* Validates trades against constraints: *Is the risk-reward ratio acceptable? Has the maximum daily drawdown been hit?* 

### F. Explainable AI & LLM Layer
* **`analyst.py` / `llm_service.py`**: The defining feature of TradeAI is its explainability. 
* Powered by an LLM integration via **Groq** (running highly capable models such as `llama-3.3-70b-versatile`), the AI takes the raw metrics, market regime, confidence scores, and calculated risk factors to synthesize human-readable insights.
* **Retrieval-Augmented Generation (RAG)**: Uses internal documentation or scraped financial concepts to further bolster the intelligence and reasoning in the agent's chat explanations.

---

## 3. Request Lifecycle / Data Flow

1. **User Request**: The user selects `EUR/USD` on the `1h` timeframe and clicks "Analyze" on the dashboard.
2. **Data Pull**: The Django backend triggers `ModelInference`. If not cached, the `TwelveDataClient` pulls the latest 1-hour candles.
3. **Feature Construction**: `processing.py` turns the raw OHLCV data into a highly dimensional sequence of indicators.
4. **Prediction**: The model corresponding to `EUR/USD 1h` is retrieved from the `ModelRegistry`. The data array is fed into the Tensorflow LSTM model which outputs a directional probability.
5. **Risk Evaluation**: The prediction and its confidence score are pushed to `RiskManager`. The system generates a Stop Loss, Take Profit, evaluates the trade setup, and returns a final `should_trade` boolean and trade parameters.
6. **LLM Synthesis**: The entire context (technical indicators, prediction, confidence, user prompt, and calculated SL/TP) is passed to the LLM agent (`ForexAnalyst`), which is strongly typed using Pydantic schemas to return a structured JSON response including a human-readable argument.
7. **Response**: The user interface renders exactly what the model predicts, risk parameters, and an intelligently worded explanation, empowering the trader.

---

## 4. Key Strengths & Potential

* **Complete Modularity**: By saving explicit `.joblib` or `.h5` model states per pair and timeframe, the system allows for independent retraining and tuning. One underperforming pair doesn't taint the rest of the ecosystem.
* **Highly Explainable**: Raw neural networks operate effectively as "black boxes." TradeAI's integration of prompt-engineered LLMs dissects the technical conditions accompanying the model signal, enforcing a "glass-box" approach so the user isn't trading blindly.
* **Production-Oriented Risk Controls**: Adding an explicit layer that filters trades if expected Risk-Reward is unreasonable or if maximum daily drawdown is achieved shows this is built beyond a proof-of-concept; it is aiming toward production safety.
