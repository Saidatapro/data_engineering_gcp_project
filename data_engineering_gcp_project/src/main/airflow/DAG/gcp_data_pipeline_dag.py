"""
gcp_data_pipeline_dag.py
========================

This Airflow DAG orchestrates the end‑to‑end data engineering workflow
for an example analytics pipeline running on Google Cloud.  It is
intended to be deployed in Cloud Composer or an Astronomer managed
Airflow environment.  The DAG executes a series of PySpark
transformation jobs on raw data (JSON, CSV and XML), trains a simple
machine learning model and generates aggregate metrics for reporting.

Dependencies among tasks reflect a typical staging pipeline: raw data
are cleansed and written to a processing zone; the model consumes
processed data and outputs artifacts to a curated zone; metrics are
produced at the end so that dashboards can be updated.

Before deploying this DAG ensure that the environment has access to
Spark (e.g., via Dataproc or a Spark on Kubernetes cluster), Google
Cloud Storage and that the necessary Python dependencies (pyspark,
google‑cloud‑storage, etc.) are installed.  The transformation
functions live in the ``scripts`` package under ``src/main/python``.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import transformation functions.  Using relative imports ensures that
# Airflow can locate the scripts when this DAG is packaged with
# Astronomer.  Depending on your deployment you might need to adjust
# the import path (e.g., using a PythonPackageOperator or BashOperator).
from src.main.python.scripts.transform_json import run_transform as transform_json
from src.main.python.scripts.transform_csv import run_transform as transform_csv
from src.main.python.scripts.transform_xml import run_transform as transform_xml
from src.main.python.scripts.train_ml_model import run_training as train_model
from src.main.python.scripts.generate_metrics import run_metrics


default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email": ["data-team@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="gcp_data_pipeline",
    default_args=default_args,
    description="End‑to‑end data pipeline on GCP with PySpark and Airflow",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["gcp", "spark", "ml"],
) as dag:
    # Define wrapper functions for Airflow tasks.  Each wrapper resolves
    # file paths using environment variables so that the DAG remains
    # flexible across environments.  The transformation functions
    # themselves handle logging and Spark session management.

    def transform_json_task(**context):
        raw_path = os.environ.get(
            "RAW_JSON_PATH",
            os.path.join(os.getcwd(), "data", "raw", "sample_orders.json"),
        )
        processing_path = os.environ.get(
            "PROCESSING_ORDERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "orders"),
        )
        transform_json(raw_path, processing_path)

    def transform_csv_task(**context):
        raw_path = os.environ.get(
            "RAW_CSV_PATH",
            os.path.join(os.getcwd(), "data", "raw", "customers.csv"),
        )
        processing_path = os.environ.get(
            "PROCESSING_CUSTOMERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "customers"),
        )
        transform_csv(raw_path, processing_path)

    def transform_xml_task(**context):
        raw_path = os.environ.get(
            "RAW_XML_PATH",
            os.path.join(os.getcwd(), "data", "raw", "products.xml"),
        )
        processing_path = os.environ.get(
            "PROCESSING_PRODUCTS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "products"),
        )
        transform_xml(raw_path, processing_path)

    def train_model_task(**context):
        orders_path = os.environ.get(
            "PROCESSING_ORDERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "orders"),
        )
        customers_path = os.environ.get(
            "PROCESSING_CUSTOMERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "customers"),
        )
        products_path = os.environ.get(
            "PROCESSING_PRODUCTS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "products"),
        )
        curated_dir = os.environ.get(
            "CURATED_DIR",
            os.path.join(os.getcwd(), "data", "curated"),
        )
        train_model(orders_path, customers_path, products_path, curated_dir)

    def generate_metrics_task(**context):
        orders_path = os.environ.get(
            "PROCESSING_ORDERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "orders"),
        )
        customers_path = os.environ.get(
            "PROCESSING_CUSTOMERS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "customers"),
        )
        products_path = os.environ.get(
            "PROCESSING_PRODUCTS_PATH",
            os.path.join(os.getcwd(), "data", "processing", "products"),
        )
        metrics_dir = os.environ.get(
            "METRICS_DIR",
            os.path.join(os.getcwd(), "data", "curated", "metrics"),
        )
        run_metrics(orders_path, customers_path, products_path, metrics_dir)

    # Create task instances
    t_json = PythonOperator(
        task_id="transform_json",
        python_callable=transform_json_task,
    )
    t_csv = PythonOperator(
        task_id="transform_csv",
        python_callable=transform_csv_task,
    )
    t_xml = PythonOperator(
        task_id="transform_xml",
        python_callable=transform_xml_task,
    )
    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )
    t_metrics = PythonOperator(
        task_id="generate_metrics",
        python_callable=generate_metrics_task,
    )

    # Define dependencies
    [t_json, t_csv, t_xml] >> t_train >> t_metrics