FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG SECRET_KEY
ARG GROQ_API_KEY
ARG TWELVE_DATA_API_KEY

ENV SECRET_KEY=${SECRET_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV TWELVE_DATA_API_KEY=${TWELVE_DATA_API_KEY}

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

CMD ["gunicorn", "AgentTrader.wsgi:application", "--bind", "0.0.0.0:8000"]
