FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mlflow[extras]==3.10.0" psycopg2-binary boto3
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
