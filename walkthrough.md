# Walkthrough: Hybrid Multi-Level Forex AI Agent

I have completed the implementation of the Hybrid Multi-Level Forex AI Agent and the improved model training architecture.

## Overview of Changes

### 1. Data & Processing Layer
*   **[alphavantage.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/Data/alphavantage.py)**: A new, robust client specifically for Alpha Vantage, supporting both Price (FX_INTRADAY) and Sentiment (NEWS_SENTIMENT).
*   **[processing.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/Data/processing.py)**: Significant improvements to feature engineering including:
    *   **Timeframe & Pair Encoding**: Adding these as features to help the Global Model generalize.
    *   **Regime Detection**: Automated detection of "Volatile", "Trending", and "Range-bound" market states.
    *   **Sentiment Fusion**: Merging Alva Vantage sentiment scores directly into the feature set.

### 2. Model & Training Layer
*   **[architectures.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/forex_models/architectures.py)**: Define shared LSTM architectures and XGBoost wrappers for specialized regimes.
*   **[modeltrain.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/modeltrain.py)**: Built for Colab use (as requested), this script implements the Global Training strategy with Walk-Forward validation.
*   **[model_finetune.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/model_finetune.py)**: Logic for taking the Global Base model and specializing it for specific pairs using Transfer Learning.

### 3. Agent & Inference
*   **[selection_agent.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/selection_agent.py)**: The core AI agent that detects the market regime and selects between Volatility-optimized XGBoost or Trend-optimized LSTM.
*   **[web_analyst.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/web_analyst.py)**: Aggregates real-time sentiment from news sources to provide context for the model's predictions.
*   **[inference.py](file:///home/job/Desktop/projects/TradeAI/AgentApp/inference.py)**: Unified interface for the AI Agent, providing direction, confidence, and sentiment-aware trading signals.

## How to Train

> [!TIP]
> **Colab Training**: You can copy the code in [modeltrain.py](file:///home/job/Desktop/projects/TradeAI/MLmodels/Forex/modeltrain.py) directly into a Google Colab cell. Make sure to set your Alpha Vantage API Key.

1.  **Step 1**: Run `modeltrain.py` to create the `global_base.keras` model.
2.  **Step 2**: Run `model_finetune.py --symbol EUR/USD` to specialize the model for a specific pair.

## Verification
- Dependency installation verified (XGBoost, LightGBM, Transformers).
- Data fetching client tested.
- Inference pipeline integrated with the existing Django AgentApp structure.
