.PHONY: help setup data train benchmark serve test docker-build docker-up docker-down k8s-deploy k8s-delete

PYTHON := python3.13
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

help:
	@echo "Projet Fil Rouge MLOps - Maintenance Predictive IoT"
	@echo "  setup       Installer les dependances (venv Python 3.13)"
	@echo "  data        Telecharger et preparer le dataset"
	@echo "  train       Entrainer un modele (defaut: xgboost)"
	@echo "  benchmark   Lancer le benchmark complet (5 modeles)"
	@echo "  serve       Lancer l API FastAPI en local"
	@echo "  test        Lancer les tests"
	@echo "  docker-up   Demarrer la stack Docker Compose"
	@echo "  docker-down Arreter la stack"
	@echo "  k8s-deploy  Deployer sur Kubernetes local"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true
	@echo "Setup termine. Activez le venv : source .venv/bin/activate"

data:
	$(PY) data/download_data.py
	$(PY) -m src.data.ingestion
	$(PY) -m src.data.features

train:
	$(PY) -m src.training.train --model xgboost

benchmark:
	$(PY) -m src.training.benchmark

serve:
	$(VENV)/bin/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PY) -m pytest tests/ -v --tb=short

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "MLflow     http://localhost:5000"
	@echo "API        http://localhost:8000/docs"
	@echo "Prometheus http://localhost:9090"
	@echo "Grafana    http://localhost:3000 (admin/admin)"

docker-down:
	docker compose down -v

k8s-deploy:
	kubectl apply -k k8s/overlays/dev/

k8s-delete:
	kubectl delete -k k8s/overlays/dev/ --ignore-not-found
