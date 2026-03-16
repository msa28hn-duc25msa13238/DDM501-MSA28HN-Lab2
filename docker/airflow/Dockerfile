FROM apache/airflow:2.8.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN python -m pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.0/constraints-3.8.txt" \
    mlflow-skinny==2.9.2 \
    numpy==1.24.4 \
    scipy==1.10.1 \
    matplotlib==3.7.5 \
    scikit-surprise==1.1.3
