# AI Forex Agent & Hybrid Multi-Level Modeling Strategy

This plan outlines the implementation of a robust AI agent for forex analysis and a hybrid multi-level model training architecture as requested.

## User Review Required

> [!IMPORTANT]
> **Alpha Vantage API Key**: You will need to provide an Alpha Vantage API key for historical data and sentiment analysis.
> **Computation Resources**: Training the "Global Model" on millions of rows (M1/M5/H1 data) is computationally expensive. It is recommended to run this on a GPU-enabled machine.

## Proposed Changes

### 1. Data Layer (`MLmodels/Forex/Data/`)

#### [NEW] [alphavantage.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/Data/alphavantage.py)
A robust client for Alpha Vantage API:
- `get_forex_history(symbol, timeframe, size="full")`
- `get_news_sentiment(tickers=None, topics=None, limit=200)`
- Native handling of rate limiting (wait/retry).

#### [MODIFY] [processing.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/Data/processing.py)
Extend feature engineering:
- **Technical Indicators**: Add and refine RSI, MACD, Bollinger Bands, ATR.
- **Sentiment Features**: Integrate sentiment scores (neutral, bullish, bearish) into the feature set.
- **Timeframe Encoding**: Add `timeframe` as a categorical feature (M1:0, M5:1, ...).
- **Pair Encoding**: Add `pair` as a categorical feature for the Global Model.
- **Improved Targets**: Implement targets for Direction (Up/Down) and Next N Period Returns.

### 2. Model Layer (`MLmodels/Forex/forex_models/`)

#### [NEW] [architectures.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/forex_models/architectures.py)
Define model classes:
- **GlobalLSTM**: Multi-layer LSTM designed for transfer learning.
- **XGBoostHybrid**: XGBoost model that can take LSTM embeddings or predictions as features.

#### [MODIFY] [modeltrain.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/modeltrain.py)
Replace current basic script with improved logic:
- **Step 1: Global Training**: Train on all available currency pairs and timeframes.
- **Step 2: Walk-Forward Validation**: Implement chronological sliding window validation (no random splits).
- **Step 3: Checkpointing**: Save the "Base Model" for fine-tuning.

#### [MODIFY] [model_finetune.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/model_finetune.py)
- Load Global Model.
- Freeze base layers (optional) and train specializing "Adapter" layers for specific pairs (e.g., EUR/USD).

### 3. Agent Layer (`AgentApp/`)

#### [NEW] [selection_agent.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/selection_agent.py)
The core "intelligence" of the system:
- **Regime Detection**: Detects if market is "Trending", "Range-bound", or "High Volatility".
- **Model Selector**: 
  - If high volatility -> XGBoost.
  - If trending -> LSTM.
  - Returns prediction, confidence, and explanation.

#### [MODIFY] [inference.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/inference.py)
Integrate the `selection_agent` into the main inference pipeline so that real-time queries benefit from the hybrid architecture.

### 4. Search & Sentiment Agent

#### [NEW] [web_analyst.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/web_analyst.py)
Integrate sentiment aggregation:
- Fetch news via Alpha Vantage.
- Optional: Scrape additional sources if API is insufficient.
- Perform aggregate analysis to provide context for the model's predictions.

---

## Open Questions

> [!NOTE]
> 1. **Data Backfill**: Should I implement a script to backfill historical data into a local database (PostgreSQL) for faster training iterations?
> 2. **Execution Targets**: For the "Return over next N candles" target, what is your preferred N? (e.g., 5 candles, 12 candles?)
> 3. **Model Selection Logic**: Do you have specific thresholds for "High Volatility" (e.g., ATR > 2.0 * SMA(ATR, 20))?

## Verification Plan

### Automated Tests
- `pytest` for Alpha Vantage client (mocking API responses).
- `pytest` for data processing (checking indicator calculations).
- Validation script to check model training logs (Walk-forward accuracy).

### Manual Verification
1. Run `python MLmodels/Forex/modeltrain.py` with a small subset of data to verify the Global training loop.
2. Initialize the `ModelSelectionAgent` and pass dummy market regimes to ensure it selects the correct architecture.
3. Test a real-time query through the AgentApp (e.g., "Analyze EUR/USD H1") and verify that sentiment data is included in the output.
