# TradeAI — Data Privacy Policy

**Effective Date:** February 24, 2026  
**Last Updated:** March 27, 2026  
**Application:** TradeAI — AI-Powered Forex Trading Intelligence Platform

---

## 1. Introduction

TradeAI is committed to protecting the privacy and security of our users. This Data Privacy Policy explains what personal data we collect, how we use it, how we store and protect it, and your rights regarding that data.

By creating an account or using the TradeAI platform, you agree to the practices described in this policy.

---

## 2. Data We Collect

### 2.1 Account & Registration Data

When you sign up for TradeAI, we collect the following personal information:

| Data Field       | Purpose                                      | Required |
| ---------------- | -------------------------------------------- | -------- |
| Username         | Unique identifier for your account           | Yes      |
| First Name       | Personalization of your experience            | Yes      |
| Email Address    | Account recovery, notifications               | Yes      |
| Date of Birth    | Age verification and regulatory compliance    | Yes      |
| Password         | Account authentication (stored as a hash)     | Yes      |

### 2.2 Trading & Interaction Data

When you use the platform, we collect:

- **Chat Prompts:** Messages you send to the AI assistant (e.g., "Analyze EUR/USD on 30min timeframe") are processed in real time to generate trading insights.
- **Trade Signals:** When a trade signal is generated, we store the signal type (BUY/SELL), currency pair, timeframe, entry price, take-profit (TP), and stop-loss (SL) levels, linked to your account.

### 2.3 Technical & Session Data

- **Session Cookies:** Django session cookies are used to maintain your authenticated state. These are essential for the platform to function and are not used for tracking.
- **CSRF Tokens:** Cross-Site Request Forgery tokens are used to protect form submissions.

### 2.4 Market Data (Non-Personal)

- We fetch publicly available Forex market data (OHLCV — Open, High, Low, Close, Volume) from the **TwelveData API** for the sole purpose of running our machine-learning models. This data does not contain any personal information.

---

## 3. How We Use Your Data

| Purpose                          | Legal Basis              | Data Used                       |
| -------------------------------- | ------------------------ | ------------------------------- |
| Account creation & authentication | Contractual necessity    | Username, email, password       |
| AI-powered trade analysis        | Contractual necessity    | Chat prompts, market data       |
| Trade signal history             | Legitimate interest      | Trade signals linked to account |
| Age verification                 | Legal obligation         | Date of birth                   |
| Platform security (CSRF, sessions) | Legitimate interest    | Session cookies, CSRF tokens    |

We do **not** use your data for:

- Advertising or marketing profiling
- Sale to third parties
- Automated decision-making that produces legal effects (trade signals are informational only and do not constitute financial advice)

---

## 4. Third-Party Services

TradeAI integrates with the following third-party services. Your data is shared with them only as described below:

### 4.1 Groq API (LLM Provider)

- **Data Shared:** Your chat prompts and AI-generated market analysis context are sent to Groq's API to generate natural-language trading insights.
- **Model Used:** Llama 3.3 70B Versatile
- **Data Retention by Groq:** Subject to [Groq's Privacy Policy](https://groq.com/privacy-policy/).
- **Note:** We do not send your personal information (name, email, date of birth) to Groq — only the content of your trading queries and the associated market analysis data.

### 4.2 TwelveData API (Market Data Provider)

- **Data Shared:** Only the requested currency pair and timeframe parameters.
- **No Personal Data** is transmitted to TwelveData.
- **Data Retention by TwelveData:** Subject to [TwelveData's Privacy Policy](https://twelvedata.com/privacy-policy).

### 4.3 Database Hosting

- **Provider:** PostgreSQL (self-hosted in development; Render PostgreSQL in production).
- **Data Stored:** All account data, trade signals, and session information.
- **Encryption:** Database credentials are stored in environment variables and are not hard-coded in the application source.

---

## 5. Data Storage & Security

### 5.1 Password Security

- All passwords are hashed using Django's default password hashing algorithm (PBKDF2 with SHA-256) before storage. We **never** store plaintext passwords.

### 5.2 Authentication & Session Security

- Django's built-in session framework manages user sessions.
- CSRF protection is enabled on all form submissions.
- `XFrameOptionsMiddleware` is active to prevent clickjacking attacks.
- `SecurityMiddleware` enforces security headers.

### 5.3 API Key Protection

- All API keys (Groq, TwelveData) and secrets (Django SECRET_KEY, database credentials) are stored in environment variables using `python-decouple` and are **never** committed to version control.

### 5.4 Data Location

- In development: Data is stored locally on the server running the Django application.
- In production: Data is stored on the configured PostgreSQL instance (e.g., Render).

---

## 6. Data Retention

| Data Type          | Retention Period                                           |
| ------------------ | ---------------------------------------------------------- |
| Account data       | Retained until you delete your account                     |
| Trade signals      | Retained for account lifetime for historical reference     |
| Chat prompts       | Processed in real time; not persistently stored server-side |
| Session cookies    | Expire based on Django session settings (default: 2 weeks) |
| Market data (OHLCV)| Not stored — fetched on demand from TwelveData             |

---

## 7. Your Rights

Depending on your jurisdiction, you may have the following rights:

- **Right of Access:** Request a copy of the personal data we hold about you.
- **Right to Rectification:** Request correction of inaccurate personal data.
- **Right to Erasure ("Right to be Forgotten"):** Request deletion of your account and all associated data.
- **Right to Data Portability:** Request your data in a machine-readable format.
- **Right to Withdraw Consent:** Where processing is based on consent, you may withdraw it at any time.

To exercise any of these rights, please contact us at the email address provided in Section 10.

---

## 8. Children's Privacy

TradeAI is not intended for individuals under the age of 18. We collect date of birth during registration to verify that users meet this age requirement. If we become aware that we have collected data from a user under 18, we will promptly delete their account and associated data.

---

## 9. Disclaimer — Not Financial Advice

All trade signals, predictions, and analyses provided by TradeAI are **for informational purposes only** and do **not** constitute financial advice, investment recommendations, or solicitation to trade. Users are solely responsible for their own trading decisions. Past model performance metrics displayed on the platform do not guarantee future results.

---

## 10. Contact Information

If you have questions, concerns, or requests regarding this Data Privacy Policy, please contact us at:

- **Email:** [privacy@tradeai.com](mailto:privacy@tradeai.com)
- **Project Repository:** [TradeAI on GitHub](https://github.com/your-username/TradeAI)

---

## 11. Changes to This Policy

We may update this Data Privacy Policy from time to time. Any changes will be posted on this page with an updated "Last Updated" date. Continued use of the platform after changes constitutes acceptance of the revised policy.

---

*© 2026 TradeAI. All rights reserved.*
