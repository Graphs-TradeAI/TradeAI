# TradeAI

TradeAI is a modular, explainable, production-ready AI platform for Forex trading analysis and prediction. It combines deep learning, feature engineering, and LLM-powered explainability in a modern Django web app.

## Features

- **Multi-timeframe, multi-pair support:** EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, and more.
- **Modular ML pipeline:** Data loading, feature engineering (RSI, MACD, EMA, etc), LSTM/hybrid models, per-pair/timeframe.
- **Incremental retraining & backtesting:** Update models with new data, simulate historical performance.
- **LLM explainability:** Gemini via LangChain interprets model outputs and generates human-readable trading insights.
- **Modern web UI:** Django-based, with dropdowns for pair/timeframe, chat-style prompt, and structured analysis output.
- **API endpoints:** For chat, prediction, and backtesting.
- **Docker & Kubernetes ready:** Production deployment with Postgres, Redis, Gunicorn, and scalable cloud support.

## System Overview

See [DEPLOYMENT.md](DEPLOYMENT.md) for a full architecture, training/inference pipeline, and deployment guide.

**Key components:**

- Data Layer: Twelve Data API client for OHLCV, multi-timeframe, paginated.
- Model Layer: Per-pair/timeframe LSTM/hybrid models, dynamic loading.
- Training Pipeline: Modular, supports incremental retraining and backtesting.
- Inference Layer: Loads correct model/scaler, preprocesses, predicts.
- LLM Layer: Gemini via LangChain for explainable, structured insights.
- Interface: Django web UI and REST API.

## Quickstart

### Local/Docker

1. Clone the repo and set up `.env` with your API keys and DB credentials.
2. Build and run:
   ```bash
   docker-compose up --build
   ```
3. Access at [http://localhost:8000](http://localhost:8000)

### Model Training & Fine-tuning

- Train a model:
  ```bash
  python -m MLmodels.Forex.training.trainer --symbol "EUR/USD" --timeframe "1h"
  ```
- Fine-tune:
  ```bash
  python -m MLmodels.Forex.model_finetune --symbol "GBP/USD" --timeframe "15min" --epochs 30
  ```
- Incremental retrain:
  ```bash
  python -m MLmodels.Forex.training.incremental --symbol "USD/JPY" --timeframe "1h" --lookback_days 30
  ```

### Web UI Usage

1. Log in or sign up.
2. Select a currency pair and timeframe from the dropdowns.
3. Enter a prompt or click "Analyze" to get a prediction and structured, explainable analysis.

### API Usage

- `/api/chat/` — Chat-style prompt + prediction
- `/api/predict/` — Structured prediction
- `/api/backtest/` — Backtesting

## Metrics Example

| Instrument Type         | Avg Price | Validation MAE | Relative Accuracy |
| ----------------------- | --------- | -------------- | ----------------- |
| Low-price FX (EUR/USD)  | ~1.0      | ~0.003–0.004   | ~99.6–99.7%       |
| High-price FX (USD/JPY) | ~150      | ~0.30–0.40     | ~99.7–99.8%       |

## Conclusion

- Models generalize well across low- and high-priced FX instruments
- High relative accuracy (~99.7%) confirms strong price prediction
- Higher timeframe models provide a usable directional edge
- Lower timeframe models are better suited for smoothing and confirmation, not direct entry signals

---

For full details on architecture, training, inference, and deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).
