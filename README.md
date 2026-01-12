# TradeAI 

TradeAI is an advanced AI-powered Forex trading intelligence platform designed to provide real-time market analysis and predictions. It leverages machine learning models to analyze currency pairs and offer actionable insights.

## Features

Intelligent Market Analysis: AI-driven predictions for major Forex pairs:
    *   EUR/USD
    *   GBP/USD
    *   USD/JPY
*   Interactive Dashboard: A chat-based interface where users can ask for specific market insights and receive data-backed responses.
*   Multi-Timeframe Analysis: Support for 15-minute, 30-minute, 1 hour, 2 hour and 4-hour timeframes.
*   Modern Landing Page: A sleek, dark-themed landing page showcasing features, testimonials, and pricing.
*   User Authentication: Secure signup and login functionality to protect user data.

##  Tech Stack

*   Backend: Python, Django
*   Frontend: HTML, CSS, JavaScript
*   AI/ML: LSTM, llm(Groq-llama model)
*   Database:  PostgreSQL
Data Sources
*   TwelveData Api
*   
##  Installation
The link: https://tradeai-v85y.onrender.com/

##  Usage

1.  Open your browser enter the link `https://tradeai-v85y.onrender.com/`.
2.  Landing Page: Explore the features and pricing.
3.  Sign Up/Login: Create an account to access the dashboard.
4.  Dashboard: Once logged in, use the chat interface to request market analysis (e.g., "Analyze EUR/USD on 30min timeframe").

## Approach
fetched the data using the TwelveData API and perfomed feature engineering to add more concrete features. Examples of features added were: volatility_10,volatility_20 etc.
The feature-rich data was fed into an LSTM with 4 layers, and a sigmoid (activation function) was added.
Model trained for 30 epochs but Early Stopping enabled early process termintaion and best model saved.

## Metrics
| Instrument Type         | Avg Price | Validation MAE | Relative Accuracy |
| ----------------------- | --------- | -------------- | ----------------- |
| Low-price FX (EUR/USD)  | ~1.0      | ~0.003–0.004   | ~99.6–99.7%       |
| High-price FX (USD/JPY) | ~150      | ~0.30–0.40     | ~99.7–99.8%       |

## conclusion
1.   Models generalize well across low- and high-priced FX instruments
2.   High relative accuracy (~99.7%) confirms strong price prediction
3.   Higher timeframe models provide a usable directional edge
4.   Lower timeframe models are better suited for smoothing and confirmation, not direct entry signals



