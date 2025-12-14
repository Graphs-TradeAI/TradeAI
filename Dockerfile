FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG SECRET_KEY
ARG GROQ_API_KEY
ARG TWELVE_DATA_API_KEY
ARG DEBUG
ARG ALLOWED_HOSTS
ARG DATABASE_URL

ENV SECRET_KEY=${SECRET_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV TWELVE_DATA_API_KEY=${TWELVE_DATA_API_KEY}
ENV DEBUG=${DEBUG}
ENV ALLOWED_HOSTS=${ALLOWED_HOSTS}
ENV DATABASE_URL=${DATABASE_URL}

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

CMD ["sh", "-c", "python manage.py collectstatic --noinput && gunicorn AgentTrader.wsgi:application --bind 0.0.0.0:8000"]
