# TradeAI System Architecture & Codebase Integration Report

TradeAI is a sophisticated, AI-driven algorithmic trading platform built with a clear separation of concerns. The system merges traditional financial feature engineering, deep learning (LSTM) for price prediction, and a Retrieval-Augmented Generation (RAG) powered Large Language Model (LLM) for plain-text explainability.

From a high-level perspective, the system is split into two primary layers loosely coupled via service classes:
1. **The Django Full-Stack Web Application (`AgentApp`/`AgentTrader`)**
2. **The Core ML & Backtesting Engine (`Forex`)**

Below is a detailed breakdown of how these components work internally and seamlessly coordinate with each other.

---

## 1. Core ML & Financial Engine (`/Forex`)

The `Forex` module contains all the raw analytical horsepower. It operates independently from the Web layer to maintain stability and reusability.

### A. Data Ingestion & State
* **`config.yaml`**: The single source of truth for the engine. It dictates sequence lengths for LSTM training, active indicators (e.g., specific EMA sliding windows, Ichimoku parameters, RSI lookbacks), and strict risk tolerances (e.g., maximum drawdown limit 15%, 1% risk per trade).
* **`client.py` & `twelvedata.py`**: Interacts directly with the TwelveData API. It is capable of fetching heavily paginated historical candlestick arrays and normalising the timestamps for downstream analysis.

### B. Feature Engineering
* **`indicators.py`**: A purely functional, highly mathematical pipeline powered by the `ta` library. The `build_features` method maps sequential indicators (MACD, Bollinger Bands, ATR, Ichimoku, EMAs 12/26/50/200) onto the DataFrame.
* **`preprocessor.py` / `processing.py`**: Historically formatted the targets, applying normalizations (e.g., turning raw prices into log-returns or fractional differentials) crucial for shielding the LSTM from vanishing gradients due to raw dollar scaling.
* **`get_indicator_snapshot`**: An adapter method that selectively strips the final row's features from the computed datasets to serve a lightweight dictionary representation over to the Django Inference layer.

### C. Deep Learning & Prediction
* **`modeltrain.py`**: Constructs time-series sequence batches (e.g., 60-candle lookbacks). Implements a deeply structured `tf.keras.Sequential` LSTM layout (using Dropout and Dense layers mapped defensively via a Huber loss function). This handles dynamic model creation for different currencies/timeframes.
* **`registry.py`**: The state manager for ML weight artifacts. It persists trained `.keras` files, matching scaling configurations, and metadata under a structured `/Models/{Symbol}/{Timeframe}` hierarchy.
* **`predictor.py`**: Bridges realtime incoming data with `registry.py`. It calls `client.py` for fresh data, runs them through `indicators.py`, injects sequence padding, queries the `.keras` model array, and returns the raw classification output/confidence score.

### D. Risk & Backtesting
* **`risk_engine.py` / `risk_manager.py`**: Accepts the raw signal from `predictor.py` alongside the active user's total equity and filters it heavily. If a model predicts 'BUY' but volatility is too low or drawdown is hit, the Risk Engine overrides the trade. It calculates Stop Losses (SL), Take Profits (TP) dynamically using ATR-multiples, and calculates Risk-Reward rules.
* **`engine.py` / `backtester.py`**: Iterates through long segments of time, triggering `predictor.py` at historical intervals implicitly verifying prediction distributions (translating individual `BUY`/`SELL` labels to aggregated F1-scores, Sharp Ratios, Directional Accuracy, and PnL).

---

## 2. Web, Gateway, & Interaction Layer (`/AgentApp`)

The web UI and LLM endpoints act as a robust Django chassis routing user requests down into the ML engine dynamically.

### A. Core Routing & Request Handlers
* **`views.py`**: Receives frontend traffic natively over JSON endpoints (`/api_predict`, `/api_chat`, `/api_backtest`). This layer handles authentication and HTTP exception boundaries, ensuring corrupt frontend calls do not leak deeply into the ML scripts.
* **`models.py`**: Manages the SQLite/PostgreSQL Django ORM for users, saving persistent `Trade` logging, history, tracking automated constraints via `TraderProfile`.

### B. The Engine Adapter: `inference.py`
This class acts as the crucial adapter between the Django synchronous request lifecycle and the heavy processing of the `Forex` module.
* When `views.py` calls `inference_service.predict()`, `inference.py` actively unifies four different domains:
  1. Bootstraps the `ForexPredictor` class.
  2. Bootstraps the `RiskEngine`.
  3. Formats the data and builds an integrated signal mapped comprehensively across risk sizes, LLM-ready variables, and directional predictions.
  4. Wraps `engine.py` backtesting and cleanly passes metrics back up into `displays metrics` properties across the dashboard landing.

### C. Explaining Decisions (The LLM / RAG Matrix)
* **`llm_service.py`**: Connects via API keys securely to Groq (running LLaMA infrastructure). When users prompt questions OR request a trade explanation, `llm_service.py` receives the finalized payload unified by `inference.py`.
* **`rag_service.py`**: A vector-embedded abstraction layer. While the classical ML system issues numerical outputs (`0.65` Confidence / `ema_50_ema_200_ratio`), the RAG module feeds structural context to the LLM indicating *why* such momentum overlaps imply a breakout. 
* By structuring the request through system prompts, the LLM takes mathematically structured prediction payloads (generated by the LSTM) and translates them into fluid NLP rationales displayed smoothly on the dashboard.

---

## 3. High-Level Integration Lifecycle (An End-to-End Workflow)

Suppose a user clicks "Start Analysis" for **EUR/USD** on a **1-hour** timeframe via the web UI:

1. **Frontend Request**: Javascript payload dispatches to `AgentApp/views.py` (`api_predict`).
2. **Adapter Ignition**: `views.py` initiates `AgentApp/inference.py`, initializing configurations under `settings.py` contexts.
3. **Data Pull**: `Forex/client.py` goes directly to the TwelveData API.
4. **Feature Map & AI Forward Pass**: `Forex/indicators.py` calculates overlapping EMAs/Ichimoku/CCI properties, handing the data structures directly to `Forex/predictor.py` which computes the LSTM probability maps.
5. **Safety Constraints applied**: The outcome cascades sequentially into `Forex/risk_engine.py`. Risk ratios, pip-slip limits, and drawdown ceilings are computed defining optimal exact position sizes in fractional volume.
6. **LLM Translation**: The combined context (Probability Matrix + Strict Risk Sizing) slides to `AgentApp/llm_service.py`. The LLM injects plain-text reasoning describing the convergence.
7. **JSON Response**: Unwinds everything neatly back up visually to the user's HTML canvas.

### Summary
TradeAI cleanly decouples its numerical prediction complexity, backtesting analytics, and front-end state. The modular design of `Forex/indicators.py` allows deep ML features to be continuously expanded (e.g., adding Ichimoku natively alongside LSTM mapping) while leaving `inference.py` as a stable broker that easily serves the dynamic insight down to Django consumers without breaking frontend dependencies.
