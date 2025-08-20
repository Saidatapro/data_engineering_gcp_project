# Data Engineering End‑to‑End Project on GCP

This repository contains a fully worked example of a data engineering
pipeline built on Google Cloud Platform (GCP).  It demonstrates how to
ingest heterogeneous data sources (JSON, CSV and XML), perform
transformations with PySpark, orchestrate workflows with Apache Airflow
running in Cloud Composer (or on [Astronomer](https://www.astronomer.io/)),
train a machine learning model and expose aggregated metrics for
visualisation tools such as Grafana or Kibana.  The project is
organised with multiple data zones (`raw`, `processing` and
`curated`) and uses the Parquet format for efficient storage in the
processing and curated layers.

> **Note:** The example code is provided primarily for educational
> purposes.  While it has been tested locally with PySpark, some
> modifications may be required to run on your specific GCP
> infrastructure (e.g. Dataproc vs. Kubernetes).  See the deployment
> section for guidance.


## Table of Contents

1. [Architecture](#architecture)
2. [Data Sources](#data-sources)
3. [Transformations](#transformations)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Metrics and Visualisation](#metrics-and-visualisation)
6. [Project Structure](#project-structure)
7. [Local Testing](#local-testing)
8. [Deployment on GCP](#deployment-on-gcp)
9. [Extending the Project](#extending-the-project)


## Architecture

At a high level this project implements the familiar medallion pattern
for data lakes: incoming data is staged in a **raw** zone, refined
into a **processing** (sometimes called bronze or silver) zone using
PySpark and then aggregated or modelled into a **curated** (gold) zone.
Each zone maps to a separate Google Cloud Storage bucket or
directory (represented here as folders in `./data`).

```
┌──────────┐      ┌─────────────────┐      ┌────────────────┐      ┌───────────────┐
│Raw zone  │ ───▶ │Processing zone │ ───▶ │ Curated zone   │ ───▶ │ Visualisation │
└──────────┘      └─────────────────┘      └────────────────┘      └───────────────┘
 JSON, CSV, XML     Parquet tables          ML models &          Grafana/Kibana
                    derived fields          predictions/metrics  dashboards
```

The workflow is scheduled via Airflow and can run daily (or any
interval) in Cloud Composer.  Each transformation is encapsulated in
its own script under `src/main/python/scripts`, and the DAG defined
in `src/main/airflow/DAG/gcp_data_pipeline_dag.py` wires the tasks
together.


## Data Sources

The sample data files live under `data/raw` and are intentionally
diverse to illustrate how to handle different file formats:

- **JSON (`sample_orders.json`)** – order transactions with fields such
  as `order_id`, `customer_id`, `product_id`, `quantity`, `price` and
  `order_date`.  The JSON parser infers the schema and the
  transformation script derives additional columns including `total_amount`.

- **CSV (`customers.csv`)** – customer demographics with columns like
  `first_name`, `last_name`, `email`, `signup_date` and `age`.  The
  script trims fields, constructs a `full_name` and flags customers as
  adults.

- **XML (`products.xml`)** – product metadata with elements such as
  `product_id`, `product_name`, `category` and `price`.  Because
  PySpark does not provide an XML parser out of the box, the script
  uses Python’s `xml.etree.ElementTree` to convert the XML into a
  Pandas DataFrame before handing it to Spark.

The included datasets are small for demonstration purposes.  In
practice you would configure your Airflow environment to read from
Cloud Storage (`gs://your-bucket/path/...`) or another landing zone.


## Transformations

Three transformation jobs implement the bulk of the ETL process:

| Script                          | Purpose                                                                | Derived Fields                                              |
|---------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------|
| `transform_json.py`             | Reads raw JSON order data, casts types, calculates order totals and date parts | `total_amount`, `order_year`, `order_month`, `order_day`    |
| `transform_csv.py`              | Cleanses customer CSV data, concatenates names and flags adults        | `full_name`, `signup_year`, `is_adult`                      |
| `transform_xml.py`              | Parses product XML into Spark DataFrame and flags premium products      | `is_premium`                                                |

Each script writes its output to a subfolder in the processing zone
(`data/processing/orders`, `customers` and `products` respectively)
using the Parquet format.  Parquet provides efficient columnar
storage, predicate pushdown and compression—features which are
especially beneficial when working with large datasets.


## Machine Learning Pipeline

The `train_ml_model.py` script demonstrates how to build a simple
binary classification model using PySpark MLlib.  It performs the
following steps:

1. **Data assembly** – Joins the orders, customers and products tables
   on their keys to form a unified training set.
2. **Label engineering** – Creates a `label` column indicating
   whether an order is “large” (default threshold is 50.0 currency
   units, configurable via the `LARGE_ORDER_THRESHOLD` environment
   variable).
3. **Feature engineering** – Indexes the product category into a
   numeric value, assembles numerical fields (quantity, price, age,
   category index) into a feature vector.
4. **Model training** – Fits a logistic regression model and
   evaluates it on a hold‑out test set using Area Under the ROC Curve
   (AUC).
5. **Artefact generation** – Writes predictions to
   `data/curated/predictions` and saves the fitted model to
   `data/curated/model`.  Additionally, it outputs a small JSON
   file with model metrics (AUC, record count) to
   `data/curated/metrics.json`.

While basic, this pipeline illustrates how to integrate machine
learning into a data platform.  You can extend it with more complex
models, additional features, hyper‑parameter tuning and evaluation
metrics as needed.


## Metrics and Visualisation

Dashboards are crucial for monitoring data quality and business
performance.  The `generate_metrics.py` script computes a few simple
aggregates from the processed datasets:

- Daily order counts and revenue
- Total revenue by product category
- Average customer age

These metrics are written to `data/curated/metrics/orders_metrics.json`.
To visualise them you can configure Grafana or Kibana to watch the
curated metrics directory (or publish the file to a monitoring
endpoint).  For example, you could use a Cloud Function or Pub/Sub
trigger to ship the JSON into Elasticsearch (for Kibana) or to
Prometheus (for Grafana).  The metrics file is small and human‑readable,
making it straightforward to ingest.


## Project Structure

```
data-engineering-gcp-project/
│
├── data/                     # Local representation of the lake
│   ├── raw/                 # Landing zone for unprocessed data
│   │   ├── sample_orders.json
│   │   ├── customers.csv
│   │   └── products.xml
│   ├── processing/          # Zone for cleansed, structured data (Parquet)
│   │   ├── orders/
│   │   ├── customers/
│   │   └── products/
│   └── curated/             # Zone for models, metrics and aggregates
│       ├── predictions/
│       ├── model/
│       └── metrics/
│
├── src/
│   └── main/
│       ├── airflow/
│       │   └── DAG/
│       │       └── gcp_data_pipeline_dag.py  # Airflow DAG definition
│       └── python/
│           └── scripts/
│               ├── transform_json.py
│               ├── transform_csv.py
│               ├── transform_xml.py
│               ├── train_ml_model.py
│               └── generate_metrics.py
│
├── requirements.txt         # Python dependencies (PySpark, Airflow, etc.)
├── .gitignore               # Ignore compiled/intermediate files
└── README.md                # You are here
```


## Local Testing

Although the project is meant for GCP, you can run the pipeline end‑to‑end
on your local machine with a few steps:

1. **Install dependencies** – Create a virtual environment and install
   requirements.  At minimum you’ll need [`pyspark`](https://pypi.org/project/pyspark/)
   and [`apache-airflow`](https://pypi.org/project/apache-airflow/).  You can install
   additional libraries like `mlflow` or `pandas` as needed.

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run transformations** – Execute each transformation script
   directly to populate the `data/processing` zone:

   ```bash
   python src/main/python/scripts/transform_json.py
   python src/main/python/scripts/transform_csv.py
   python src/main/python/scripts/transform_xml.py
   ```

3. **Train the model** – Run the ML script to generate a model and
   predictions:

   ```bash
   python src/main/python/scripts/train_ml_model.py
   ```

4. **Compute metrics** – Generate aggregate metrics for dashboards:

   ```bash
   python src/main/python/scripts/generate_metrics.py
   ```

5. **Inspect outputs** – You can explore the Parquet files with
   [Parquet tools](https://github.com/apache/parquet-mr/tree/master/parquet-tools) or by reading them back into Spark.

6. **Run the DAG** – To test the full workflow with Airflow, initialise
   a local Airflow instance, copy the DAG file into your DAGs folder and
   trigger it.  See the [Airflow documentation](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
   for details.


## Deployment on GCP

To deploy this project in a production‑like setting on Google Cloud you
would typically perform the following steps:

1. **Create storage buckets** for each zone (`raw`, `processing`,
   `curated`).  You might choose separate buckets or prefixes within
   a single bucket depending on your governance strategy.

2. **Provision a Cloud Composer environment** (Airflow) and ensure it
   has network access to your Spark cluster and Cloud Storage.  Cloud
   Composer 2 supports Python 3.10 and Airflow 2.x which is compatible
   with the code provided here.

3. **Configure a Spark execution environment** – This could be
   [Dataproc](https://cloud.google.com/dataproc), a managed cluster running
   [Spark on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/spark-on-gke),
   or a [Dataproc Serverless](https://cloud.google.com/dataproc-serverless)
   job.  Ensure that `pyspark` and any required libraries are
   available.  In Airflow you might use the
   `DataprocPySparkOperator` or `KubernetesPodOperator` to run the
   scripts.

4. **Package the DAG** – Use the [Astronomer CLI](https://docs.astronomer.io/astro-cli) to
   build a deployable Airflow image, including your scripts and any
   dependencies.  For example:

   ```bash
   astro dev init  # initialise an Astronomer project
   # Copy src/main/airflow/DAG/gcp_data_pipeline_dag.py into the dags/ folder
   # Copy src/main/python/scripts into the include/ folder or package them as a wheel
   # Update Dockerfile as needed to install pyspark and other libs
   astro dev start  # run locally
   ```

5. **Set environment variables** – In Cloud Composer, define
   variables (via Airflow Variables or Environment Config) for the
   paths to your buckets (e.g. `RAW_JSON_PATH`, `PROCESSING_ORDERS_PATH`,
   `CURATED_DIR`, etc.).  This allows the same codebase to run in
   multiple environments (local, staging, prod).

6. **Visualise metrics** – Point Grafana or Kibana at the
   curated metrics location.  For Kibana you might load the JSON
   into an Elasticsearch index using a Filebeat or Logstash pipeline.
   For Grafana you could expose the metrics via a custom collector or
   use [Grafana Loki](https://grafana.com/oss/loki/) for log‑based metrics.


## Extending the Project

This repository is a starting point for building production‑grade
pipelines.  Here are a few ideas for enhancements:

- **Schema management** – Define explicit Spark schemas to guard against
  unexpected changes in upstream data.  Tools like [Delta Lake](https://delta.io/)
  or [Iceberg](https://iceberg.apache.org/) can also provide ACID
  semantics and time travel.
- **Automated testing** – Add unit tests for your transformation
  functions using PyTest or Airflow's DAG validation utilities.
- **Data quality checks** – Integrate [Great Expectations](https://greatexpectations.io/)
  to validate incoming data and catch anomalies early.
- **Streaming ingestion** – Adapt the scripts to consume data from
  Pub/Sub or Kafka using Spark Structured Streaming.
- **Advanced ML** – Replace the logistic regression with more
  sophisticated models (e.g. gradient boosting, deep learning) and
  experiment with hyper‑parameter tuning frameworks such as
  Hyperopt or MLflow.

Feel free to fork this repository and tailor it to your own use cases.


## License

This project is licensed under the [MIT License](LICENSE).  You are
free to use, modify and distribute the code provided you include the
appropriate attribution.