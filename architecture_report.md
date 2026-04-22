# TradeAI Architecture Report

This document is intentionally kept lightweight to avoid drift.

For the current, validated architecture and readiness assessment, see:
- `system_analysis.md`
- `README.md`

Current canonical runtime path:
- `Forex/indicators.py` -> `Forex/modeltrain.py` -> `Forex/registry.py` -> `Forex/predictor.py` -> `Forex/risk_engine.py` -> `AgentApp/inference.py` -> `AgentApp/views.py` / UI.
