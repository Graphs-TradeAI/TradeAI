# TradeAI System Overview

## 1. System Architecture

TradeAI is a modular, production-ready forex trading agent with:

- **Data Layer:** Fetches multi-timeframe OHLCV data from Twelve Data API.
- **Model Layer:** Per-currency-pair, per-timeframe LSTM/hybrid models, saved/loaded dynamically.
- **Training Pipeline:** Modular pipeline for data loading, preprocessing, feature engineering, model training, evaluation, incremental retraining, and backtesting.
- **Inference Layer:** Loads correct model/scaler, preprocesses new data, runs prediction.
- **LLM Reasoning Layer:** Uses Gemini via LangChain to generate human-readable, explainable trading insights.
- **Interface Layer:** Django web UI with dropdowns for pair/timeframe, chat-style prompt, and structured analysis output.
- **Deployment:** Dockerized with Django, Gunicorn, Postgres, and Redis.

## 2. Model Training & Fine-tuning Pipeline

- **Data Loading:**
  - Uses `TwelveDataClient` to fetch historical OHLCV for any supported pair/timeframe.
  - Handles pagination and large datasets.

- **Preprocessing:**
  - Scaling (MinMaxScaler), windowing, and feature engineering (RSI, MACD, EMA, etc).
  - Sequence creation for LSTM input.

- **Model Training:**
  - Each (pair, timeframe) gets its own model and scaler, saved in `forex_models/{PAIR}/{TIMEFRAME}/`.
  - Training pipeline supports walk-forward splits, early stopping, and checkpointing.
  - Metrics: MAE, MSE, directional accuracy, Sharpe ratio, win rate, max drawdown.

- **Incremental Retraining:**
  - `IncrementalTrainer` loads existing model/scaler and fine-tunes on new data only (no full retrain).

- **Backtesting:**
  - Historical simulation using trained models, with full metrics and trade logs.

## 3. Inference Pipeline

- **User selects pair/timeframe (or enters prompt).**
- **System loads correct model/scaler** using ModelRegistry.
- **Fetches latest data** from Twelve Data.
- **Preprocesses and builds sequence** for prediction.
- **Runs model inference** to get direction probability and predicted price.
- **LLM Layer (Gemini via LangChain):**
  - Interprets model output, indicators, and risk.
  - Generates structured, human-readable insight (signal, confidence, reasoning, indicators used, etc).

## 4. Training & Fine-tuning Instructions

- **Train a new model:**
  - Run: `python -m MLmodels.Forex.training.trainer --symbol "EUR/USD" --timeframe "1h"`
  - Or train all: `python -m MLmodels.Forex.training.trainer --all`

- **Fine-tune a model:**
  - Run: `python -m MLmodels.Forex.model_finetune --symbol "GBP/USD" --timeframe "15min" --epochs 30`

- **Incremental retraining:**
  - Run: `python -m MLmodels.Forex.training.incremental --symbol "USD/JPY" --timeframe "1h" --lookback_days 30`

- **Backtesting:**
  - Use the Django API endpoint `/api/backtest/` or run backtest functions in the inference layer.

## 5. User Interaction

- **Web UI:**
  - User logs in and is presented with dropdowns for currency pair and timeframe.
  - User can enter a prompt/question or just click "Analyze".
  - System fetches latest data, runs prediction, and displays structured analysis (signal, confidence, reasoning, indicators, etc).
  - All results are explainable and include LLM-generated insights.

- **API:**
  - `/api/chat/` for chat-style prompt + prediction.
  - `/api/predict/` for structured prediction.
  - `/api/backtest/` for backtesting.

## 6. Deployment

- **Docker Compose:**
  - `docker-compose up --build`
  - Services: Django (web), Postgres (db), Redis (cache/queue)
  - Configure `.env` for API keys and DB credentials.

- **Production:**
  - Gunicorn serves Django app.
  - Static files collected at build.
  - All secrets managed via `.env` and Docker Compose.

---

For further details, see code comments and each module's README/docstrings.

# TradeAI Deployment Guide

## Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose installed

### Local Development with Docker Compose

1. **Build and run all services:**

   ```bash
   docker-compose up -d
   ```

2. **View logs:**

   ```bash
   docker-compose logs -f
   ```

3. **Stop services:**

   ```bash
   docker-compose down
   ```

4. **Access the application:**
   - Application: http://localhost:8000
   - With Nginx: http://localhost

### Building Docker Image

```bash
chmod +x docker-build.sh
./docker-build.sh
```

Or manually:

```bash
docker build -t tradeai:latest .
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube, GKE, EKS, AKS, etc.)
- kubectl configured
- Docker registry access

### Deployment Steps

1. **Update the image registry in k8s-deployment.yaml:**
   Replace `your-registry/tradeai:latest` with your actual registry URL.

2. **Update secrets:**
   Edit the `tradeai-secrets` section in `k8s-deployment.yaml`:
   - `SECRET_KEY`: Generate a new Django secret key
   - `DATABASE_URL`: Update if using external database
   - `POSTGRES_PASSWORD`: Change to a secure password

3. **Update ConfigMap:**
   Edit `ALLOWED_HOSTS` in the ConfigMap to include your domain.

4. **Build and push Docker image:**

   ```bash
   docker build -t your-registry/tradeai:latest .
   docker push your-registry/tradeai:latest
   ```

5. **Deploy to Kubernetes:**

   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

6. **Check deployment status:**

   ```bash
   kubectl get pods -n tradeai
   kubectl get services -n tradeai
   ```

7. **Get the external IP:**
   ```bash
   kubectl get service tradeai-service -n tradeai
   ```

### Scaling

The deployment includes a Horizontal Pod Autoscaler (HPA) that automatically scales between 2-10 replicas based on CPU and memory usage.

Manual scaling:

```bash
kubectl scale deployment tradeai-web --replicas=5 -n tradeai
```

### Monitoring

View logs:

```bash
kubectl logs -f deployment/tradeai-web -n tradeai
```

Execute commands in a pod:

```bash
kubectl exec -it deployment/tradeai-web -n tradeai -- python manage.py shell
```

### Database Migrations

Migrations run automatically via init container. To run manually:

```bash
kubectl exec -it deployment/tradeai-web -n tradeai -- python manage.py migrate
```

### Cleanup

```bash
kubectl delete namespace tradeai
```

## Environment Variables

Required environment variables (set in .env for Docker or k8s secrets):

- `SECRET_KEY`: Django secret key
- `DATABASE_URL`: PostgreSQL connection string
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `DEBUG`: Set to False in production

## Security Notes

1. **Change default passwords** in production
2. **Use proper secrets management** (e.g., Kubernetes Secrets, AWS Secrets Manager)
3. **Enable HTTPS** with cert-manager or cloud load balancer
4. **Set proper ALLOWED_HOSTS** in production
5. **Use a proper SECRET_KEY** (generate with `python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'`)

## Troubleshooting

### Docker Compose Issues

**Database connection errors:**

```bash
docker-compose down -v
docker-compose up -d
```

**Static files not loading:**

```bash
docker-compose exec web python manage.py collectstatic --noinput
```

### Kubernetes Issues

**Pods not starting:**

```bash
kubectl describe pod <pod-name> -n tradeai
kubectl logs <pod-name> -n tradeai
```

**Database connection issues:**
Check if PostgreSQL is running:

```bash
kubectl get pods -n tradeai | grep postgres
kubectl logs deployment/postgres -n tradeai
```

**Image pull errors:**
Ensure your image is pushed to the registry and credentials are configured:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n tradeai
```

Then add to deployment spec:

```yaml
imagePullSecrets:
  - name: regcred
```
