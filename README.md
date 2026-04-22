# TradeAI

TradeAI is a Django-based Forex analysis platform with:
- Config-driven multi-pair/multi-timeframe model training
- Multitask deep learning inference (return regression + direction classification)
- Risk-engine filtered trading signals (`BUY` / `SELL` / `HOLD`)
- LLM-generated reasoning on top of deterministic model/risk outputs
- Backtesting and model registry support

## Active System Architecture

### Core runtime modules
- `Forex/indicators.py`: Single source of truth for feature engineering and targets.
- `Forex/modeltrain.py`: Config-driven trainer (`pairs x timeframes`) with multitask LSTM heads:
  - `return_head` -> predicts next log return (`target_return`)
  - `direction_head` -> predicts up/down probability (`target_direction`)
- `Forex/predictor.py`: Real-time inference with strict feature-contract validation from `metrics.json`.
- `Forex/risk_engine.py`: Position sizing, SL/TP, and trade filters.
- `Forex/engine.py`: Backtest simulation using the same model contract as inference.
- `Forex/registry.py`: Model/metrics storage and model discovery.
- `AgentApp/inference.py`: Adapter that unifies predictor + risk output for APIs/UI.
- `AgentApp/views.py`: API endpoints and dashboard pages.

### Removed legacy modules
The following legacy files were removed because they were not used by the active runtime path:
- `Forex/processing.py`
- `Forex/preprocessor.py`
- `Forex/twelvedata.py`
- `Forex/backtester.py`
- `Forex/risk_manager.py`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure `.env`:
- `SECRET_KEY`
- `DEBUG`
- `ALLOWED_HOSTS`
- `TWELVE_DATA_API_KEY`
- `GROQ_API_KEY`
- Optional DB/email settings

4. Run migrations:
```bash
.venv/bin/python manage.py migrate
```

5. Start app:
```bash
.venv/bin/python manage.py runserver
```

## Training

### Train all configured pairs/timeframes
Reads `pairs` and `timeframes` from `Forex/config.yaml` and trains all combinations:
```bash
.venv/bin/python -m Forex.modeltrain
```

### Programmatic subset training
```python
from Forex.modeltrain import ForexTrainer

trainer = ForexTrainer()
trainer.train_from_config(
    pairs=["AUD/USD", "EUR/USD"],
    timeframes=["1h", "4h"],
)
```

Models and metrics are saved under `Forex/Models/<PAIR>/<timeframe>/`.

## Inference and API

### Main endpoints
- `POST /api/chat/`: legacy chat + prediction + metrics payload
- `POST /api/predict/`: structured signal + risk + reasoning
- `POST /api/backtest/`: backtest report + recent trades
- `GET /api/models/`: available trained models + recommended model

### Signal generation flow
1. Fetch latest candles
2. Build features with `indicators.py`
3. Enforce feature contract saved at training time
4. Run model:
   - Return head -> predicted return / predicted price
   - Direction head -> direction probability
5. Combine confidence and apply `HOLD` threshold
6. Run risk checks and generate trade parameters

## Validation Commands

```bash
.venv/bin/python -m compileall Forex AgentApp tests -q
DEBUG=True .venv/bin/python manage.py check
```

## Notes

- If `pytest` is not installed, `tests/` cannot be executed via pytest until it is added.
- Production readiness still depends on operational controls (monitoring, broker reconciliation, staged rollout, drift alerts), even when code-level checks pass.
