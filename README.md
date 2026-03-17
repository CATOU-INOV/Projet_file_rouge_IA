# Projet Fil Rouge MLOps – Maintenance Prédictive IoT

**Master 2 Data & IA – YNOV Campus Montpellier 2025-2026**

Système de maintenance prédictive pour turbomachines basé sur le dataset NASA C-MAPSS.
L'objectif est d'estimer la **RUL (Remaining Useful Life)** de moteurs d'avion à partir de données capteurs.

---

## Architecture

```
CSV (NASA C-MAPSS)
  └─► Ingestion & Validation (Parquet + manifest MD5)
        └─► Feature Engineering (Rolling, Delta, EMA, Normalisation)
              └─► Benchmark 5 modèles → MLflow Registry
                    └─► API FastAPI (/predict, /health, /metrics, /model/version)
                          └─► Monitoring (Prometheus + Grafana + Drift Detection)
                                └─► Docker Compose / Kubernetes (Kind/Minikube/K3s)
```

### Stack technique

| Composant       | Technologie                          |
|-----------------|--------------------------------------|
| Données         | NASA C-MAPSS FD001 (Kaggle)          |
| ML (tabulaire)  | RandomForest, XGBoost, LightGBM      |
| ML (DL)         | LSTM, CNN1D (PyTorch)                |
| Tracking ML     | MLflow + PostgreSQL + MinIO (S3)     |
| API             | FastAPI + Pydantic + Uvicorn         |
| Monitoring      | Prometheus + Grafana                 |
| Containerisation| Docker Compose                       |
| Orchestration   | Kubernetes (Kustomize + Canary)      |

---

## Lancement rapide

### Prérequis

- Python 3.13 (via Homebrew : `brew install python@3.13`)
- Docker & Docker Compose
- (optionnel) kubectl + Kind/Minikube pour K8s

### 1. Setup

```bash
make setup
cp .env.example .env
```

### 2. Données

```bash
make data
```

Télécharge `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` dans `data/raw/`,
puis exécute l'ingestion et le feature engineering. Les Parquet sont sauvegardés dans `data/features/`.

### 3. Stack complète (Docker Compose)

```bash
make docker-up
```

Services disponibles :

| Service    | URL                          | Identifiants |
|------------|------------------------------|--------------|
| API        | http://localhost:8000/docs   | –            |
| MLflow     | http://localhost:5001        | –            |
| MinIO      | http://localhost:9001        | minioadmin / minioadmin |
| Prometheus | http://localhost:9091        | –            |
| Grafana    | http://localhost:3000        | admin / admin |

### 4. Benchmark (5 modèles)

```bash
make benchmark
```

Lance RF, XGBoost, LightGBM, LSTM et CNN1D. Les métriques sont trackées dans MLflow
et un CSV comparatif est sauvegardé dans `data/features/benchmark_*.csv`.

> Résultats obtenus sur les modèles tabulaires (DL en cours de re-benchmark) :
> | Modèle | MAE | RMSE | R² | sMAPE |
> |--------|-----|------|-----|-------|
> | XGBoost | **3.07** | **4.52** | **0.988** | 5.96% |
> | LightGBM | 3.52 | 4.97 | 0.985 | 7.15% |
> | Random Forest | 4.50 | 6.60 | 0.974 | 7.70% |

### 5. Tests

```bash
make test
```

### 6. Entraîner un modèle spécifique

```bash
python -m src.training.train --model xgboost
# ou: random_forest | lightgbm | lstm | cnn1d
```

### 7. Déploiement Kubernetes (local)

```bash
# Overlay dev (1 réplica)
make k8s-deploy

# Vérification (dry-run)
kubectl apply -k k8s/overlays/dev/ --dry-run=client

# Overlay prod avec stratégie Canary (3 stable + 1 canary)
kubectl apply -k k8s/overlays/prod/
```

---

## Structure du projet

```
.
├── data/
│   ├── download_data.py        # Téléchargement NASA C-MAPSS
│   ├── raw/                    # Données brutes (train/test/RUL)
│   └── features/               # Parquet enrichis + scaler + benchmark CSV
├── src/
│   ├── config.py               # Configuration centralisée
│   ├── data/
│   │   ├── ingestion.py        # Chargement, validation, RUL, Parquet
│   │   └── features.py         # Rolling, delta, normalisation, séquences DL
│   ├── models/
│   │   ├── factory.py          # Factory Pattern (register_model / get_model)
│   │   ├── baseline.py         # Utilitaires tabulaires + métriques
│   │   └── deep_learning.py    # LSTM, CNN1D, callbacks
│   ├── training/
│   │   ├── train.py            # Pipeline MLflow (tabulaire + DL)
│   │   └── benchmark.py        # Benchmark 5 modèles → CSV
│   ├── serving/
│   │   ├── app.py              # API FastAPI (predict/health/metrics/version)
│   │   ├── schemas.py          # Schemas Pydantic
│   │   └── Dockerfile          # Multi-stage, non-root (UID 1001)
│   └── monitoring/
│       └── drift_detector.py   # KS-test + PSI
├── tests/
│   ├── test_ingestion.py       # Tests ingestion (load, RUL, validation)
│   ├── test_features.py        # Tests feature engineering
│   ├── test_models.py          # Tests factory + métriques
│   ├── test_api.py             # Tests API FastAPI
│   ├── test_schemas.py         # Tests schemas Pydantic
│   └── test_drift.py           # Tests détection de drift
├── k8s/
│   ├── base/                   # Deployment, Service, HPA, ConfigMap
│   └── overlays/
│       ├── dev/                # 1 réplica, ressources réduites
│       └── prod/               # 3 réplicas + stratégie Canary
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/       # Datasources + dashboard auto-provisioning
│       └── dashboards/         # turbofan.json (prédictions, latence)
├── docker-compose.yml          # 6 services (postgres, minio, mlflow, api, prometheus, grafana)
├── Makefile
└── requirements.txt
```

---

## API – Endpoints

### `POST /predict`

**Corps (JSON) :**
```json
{
  "unit_id": 1,
  "cycle": 50,
  "setting_1": -0.0007,
  "setting_2": -0.0004,
  "setting_3": 100.0,
  "sensor_2": 641.82,
  "sensor_3": 1589.70,
  "sensor_4": 1400.60,
  "sensor_7": 14.62,
  "sensor_8": 21.61,
  "sensor_9": 554.36,
  "sensor_11": 2388.02,
  "sensor_12": 9046.19,
  "sensor_13": 1.30,
  "sensor_14": 47.47,
  "sensor_15": 521.66,
  "sensor_17": 2388.02,
  "sensor_20": 39.06,
  "sensor_21": 23.419
}
```

**Réponse :**
```json
{
  "unit_id": 1,
  "predicted_rul": 87.34,
  "model_version": "Production",
  "model_name": "turbofan_rul_predictor"
}
```

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_version": "Production"}
```

### `GET /metrics`
Métriques Prometheus : `prediction_count_total`, `prediction_latency_seconds`, `prediction_errors_total`

### `GET /model/version`
```json
{"model_name": "turbofan_rul_predictor", "model_version": "Production", "experiment": "turbofan-rul-prediction", "tracking_uri": "http://localhost:5000"}
```

---

## Détection de Drift

```python
from src.monitoring.drift_detector import detect_data_drift, compute_psi
import numpy as np

# KS-test par feature
ref = np.random.normal(0, 1, (500, 5))
cur = np.random.normal(0.5, 1, (500, 5))
result = detect_data_drift(ref, cur, feature_names=["s2","s3","s4","s7","s8"])
print(result["any_drift"])        # True/False
print(result["drift_features"])   # liste des features en drift

# PSI (Population Stability Index)
psi_result = compute_psi(ref[:, 0], cur[:, 0])
print(psi_result["psi"])          # valeur PSI
print(psi_result["interpretation"])  # "stable" | "moderate_change" | "significant_drift"
```

---

## Critères de qualité

- [x] Code versionné sur Git avec commits clairs
- [x] `make docker-up` lance la stack complète sans erreur
- [x] L'API répond sur `/predict` avec données valides
- [x] MLflow tracke au moins 5 runs avec métriques (MAE, RMSE, R², sMAPE)
- [x] Manifestes K8s valides (`kubectl apply --dry-run=client`)
- [x] Au moins 5 fichiers de tests avec 20+ assertions
- [x] Monitoring Prometheus/Grafana fonctionnel avec dashboard auto-provisionné

---

## Portfolio

Ce projet illustre une chaîne MLOps complète, de la donnée brute au monitoring en production.

### Compétences démontrées

| Domaine | Détail |
|---|---|
| **Machine Learning** | Benchmark 5 modèles, cross-validation, feature engineering (rolling, delta, EMA) |
| **MLOps** | MLflow tracking & registry, gestion des versions, alias Production |
| **API & Serving** | FastAPI, Pydantic, Uvicorn, endpoints REST documentés |
| **Monitoring** | Prometheus, Grafana, détection de drift (KS-test + PSI) |
| **Containerisation** | Docker Compose, 6 services orchestrés, multi-stage Dockerfile |
| **Orchestration** | Kubernetes (Kustomize), stratégie Canary, HPA |
| **Qualité** | Tests unitaires & intégration, 20+ assertions, CI-ready |

### Résultats clés

- **XGBoost** : MAE = 3.07 cycles, R² = 0.988 sur le dataset NASA C-MAPSS
- Cross-validation 5-fold : MAE = 9.53 ± 0.80 (robustesse confirmée)
- Latence API p95 < 10ms
- `make docker-up` → stack complète en < 60 secondes

### Stack complète

```
Python 3.13 · XGBoost · LightGBM · PyTorch · MLflow · FastAPI
PostgreSQL · MinIO · Prometheus · Grafana · Docker · Kubernetes
```

### Reproduire le projet

```bash
git clone https://github.com/CATOU-INOV/Projet_file_rouge_IA.git
cd Projet_file_rouge_IA
make setup
make data
make docker-up
# Puis ouvrir http://localhost:8000/docs
```

### Démo rapide — dégradation d'un moteur

```bash
for cycle in 10 50 100 150 180; do
  echo -n "cycle=$cycle → "
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"unit_id\":1,\"cycle\":$cycle,\"setting_1\":-0.0007,\"setting_2\":-0.0004,\"setting_3\":100.0,\"sensor_2\":641.82,\"sensor_3\":1589.70,\"sensor_4\":1400.60,\"sensor_7\":14.62,\"sensor_8\":21.61,\"sensor_9\":554.36,\"sensor_11\":2388.02,\"sensor_12\":9046.19,\"sensor_13\":1.30,\"sensor_14\":47.47,\"sensor_15\":521.66,\"sensor_17\":2388.02,\"sensor_20\":39.06,\"sensor_21\":23.419}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'RUL={d[\"predicted_rul\"]}')"
done
```

Résultat attendu :
```
cycle=10  → RUL=125.06   ✅ moteur neuf
cycle=50  → RUL=126.08   ✅ nominal
cycle=100 → RUL=92.0     🟡 début de dégradation
cycle=150 → RUL=42.06    🟠 alerte
cycle=180 → RUL=11.98    🔴 intervention urgente
```
