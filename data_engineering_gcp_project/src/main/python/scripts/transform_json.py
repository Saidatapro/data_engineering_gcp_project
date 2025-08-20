"""
transform_json.py
==================

This module contains a simple PySpark job that reads order data in JSON
format from a raw zone and writes a transformed version of the data to
the processing zone in Parquet format.  The goal of this script is to
illustrate how a typical data engineering transformation might be built
around semi‑structured JSON data.  It extracts useful fields, derives
new values (such as a total monetary amount for each order), and
performs basic type casting.

The script is designed to be executed as part of a larger Airflow DAG
but can also be run standalone for local testing.  Input and output
paths can be overridden via environment variables which makes the code
portable across local development, Cloud Composer and other
environments.
"""

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofmonth


def run_transform(raw_path: str, output_path: str) -> None:
    """Run the JSON transformation job.

    Parameters
    ----------
    raw_path : str
        Absolute or relative path to the raw JSON file containing order data.
        The file is expected to contain an array of JSON objects.  If the
        file resides in a Google Cloud Storage bucket, the path should be
        prefaced with ``gs://``.
    output_path : str
        Destination directory where the processed Parquet files will be
        written.  If the directory does not exist, it will be created.

    Notes
    -----
    The function logs simple status messages to stdout so that
    progress can be monitored in Airflow and Cloud Composer logs.
    """
    spark = (
        SparkSession.builder
        .appName("Transform JSON Orders")
        .getOrCreate()
    )

    print(f"[transform_json] Reading raw JSON from {raw_path}")
    # Read the JSON file into a Spark DataFrame.  By default Spark will
    # infer the schema but you can specify one explicitly for production
    # workloads to avoid schema drift.
    orders_df = spark.read.json(raw_path)

    print("[transform_json] Performing type casting and deriving fields…")
    # Cast numeric fields to appropriate types and derive new ones.  We
    # intentionally perform explicit casts here instead of relying on the
    # inferred schema so that data types are deterministic.
    orders_df = (
        orders_df
        .withColumn("quantity", col("quantity").cast("integer"))
        .withColumn("price", col("price").cast("double"))
        .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))
        .withColumn("total_amount", col("quantity") * col("price"))
        .withColumn("order_year", year(col("order_date")))
        .withColumn("order_month", month(col("order_date")))
        .withColumn("order_day", dayofmonth(col("order_date")))
    )

    print("[transform_json] Writing transformed data to Parquet…")
    (
        orders_df
        .repartition(1)  # small dataset – single file; scale up partitions for large datasets
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    print(f"[transform_json] Successfully wrote Parquet to {output_path}")

    spark.stop()


def main() -> None:
    """Entry point for local execution.

    When run as a script, this function resolves the input and output
    paths from environment variables or defaults, invokes the
    transformation and exits.  This allows you to test the code
    outside of Airflow by simply running ``python transform_json.py``.
    """
    raw_default = os.path.join(os.getcwd(), "data", "raw", "sample_orders.json")
    processing_default = os.path.join(os.getcwd(), "data", "processing", "orders")
    raw_path = os.environ.get("RAW_JSON_PATH", raw_default)
    output_path = os.environ.get("PROCESSING_ORDERS_PATH", processing_default)
    run_transform(raw_path, output_path)


if __name__ == "__main__":
    main()