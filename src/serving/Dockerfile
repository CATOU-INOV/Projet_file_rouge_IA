# ── Stage 1 : Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Installer les outils de build nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances dans un virtualenv isolé
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2 : Image finale légère ────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dépendances runtime minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances installées depuis le builder
COPY --from=builder /install /usr/local

# Copier le code source
COPY src/ ./src/
COPY .env.example .env.example

# Créer un utilisateur non-root (UID 1001)
RUN groupadd -r appgroup && useradd -r -u 1001 -g appgroup appuser && \
    chown -R appuser:appgroup /app

USER 1001

EXPOSE 8000

# HEALTHCHECK : vérification de liveness toutes les 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
