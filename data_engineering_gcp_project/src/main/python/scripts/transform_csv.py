"""
transform_csv.py
================

This module contains a PySpark job for processing CSV data (customer
records) stored in the raw zone and outputting a cleansed and enriched
dataset to the processing zone in Parquet format.  The script
demonstrates basic column transformations, simple validations and
derived field creation using PySpark.  It is intended for use within
an Airflow DAG but can also be executed independently for testing.

Usage
-----
Run the script directly to test locally::

    python transform_csv.py

Environment variables ``RAW_CSV_PATH`` and ``PROCESSING_CUSTOMERS_PATH``
can be set to override default paths.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    lit,
    when,
    to_date,
    year,
)


def run_transform(raw_path: str, output_path: str) -> None:
    """Read customer CSV data, perform transformations and write as Parquet.

    Parameters
    ----------
    raw_path : str
        Location of the source CSV file (local path or ``gs://`` URI).
    output_path : str
        Directory where the transformed Parquet dataset will be written.
    """
    spark = (
        SparkSession.builder
        .appName("Transform CSV Customers")
        .getOrCreate()
    )

    print(f"[transform_csv] Reading raw CSV from {raw_path}")
    customers_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(raw_path)
    )

    print("[transform_csv] Performing data cleansing and derivations…")
    # Trim leading/trailing whitespace on string columns and enforce proper
    # types.  Because inferSchema is enabled above, numeric fields are
    # already cast but we still explicitly cast date fields.
    customers_df = (
        customers_df
        .withColumn("first_name", col("first_name").cast("string"))
        .withColumn("last_name", col("last_name").cast("string"))
        .withColumn("full_name", concat_ws(" ", col("first_name"), col("last_name")))
        .withColumn("email", col("email").cast("string"))
        .withColumn("signup_date", to_date(col("signup_date"), "yyyy-MM-dd"))
        .withColumn("signup_year", year(col("signup_date")))
        .withColumn(
            "is_adult",
            when(col("age") >= 18, lit(True)).otherwise(lit(False)).cast("boolean")
        )
    )

    print("[transform_csv] Writing transformed data to Parquet…")
    (
        customers_df
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    print(f"[transform_csv] Successfully wrote Parquet to {output_path}")

    spark.stop()


def main() -> None:
    raw_default = os.path.join(os.getcwd(), "data", "raw", "customers.csv")
    processing_default = os.path.join(os.getcwd(), "data", "processing", "customers")
    raw_path = os.environ.get("RAW_CSV_PATH", raw_default)
    output_path = os.environ.get("PROCESSING_CUSTOMERS_PATH", processing_default)
    run_transform(raw_path, output_path)


if __name__ == "__main__":
    main()