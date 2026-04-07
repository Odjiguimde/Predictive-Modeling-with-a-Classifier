# ──────────────────────────────────────────────────
# Projet 5 — Churn Predictor
# Docker : Streamlit Dashboard + Flask API
# ──────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Port Streamlit
EXPOSE 8501

# Port Flask API
EXPOSE 5000

# Lancer Streamlit par défaut
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
