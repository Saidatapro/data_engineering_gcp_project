"""
generate_metrics.py
===================

This auxiliary script computes a few simple aggregate metrics from the
processed datasets and writes them to the curated zone as JSON files
that can easily be ingested by visualization tools such as Grafana or
Kibana.  It aggregates order totals by date and category and
computes customer demographic statistics.  Additional metrics can be
added as needed; the examples provided here are intended to serve as
a template for more sophisticated reporting.
"""

import os
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as _sum, count as _count, avg as _avg, to_date


def run_metrics(orders_path: str, customers_path: str, products_path: str, metrics_dir: str) -> None:
    spark = (
        SparkSession.builder
        .appName("Generate Metrics")
        .getOrCreate()
    )

    # Load processed data
    orders_df = spark.read.parquet(orders_path)
    customers_df = spark.read.parquet(customers_path)
    products_df = spark.read.parquet(products_path)

    # Join to enrich orders with category
    joined_df = (
        orders_df
        .join(products_df.select("product_id", "category"), on="product_id", how="left")
        .join(customers_df.select("customer_id", "age", "signup_year"), on="customer_id", how="left")
    )

    print("[generate_metrics] Computing order totals by date…")
    daily_totals = (
        joined_df
        .groupBy(to_date("order_date").alias("order_date"))
        .agg(
            _count("order_id").alias("order_count"),
            _sum("total_amount").alias("total_revenue")
        )
        .orderBy("order_date")
    ).toPandas().to_dict(orient="records")

    print("[generate_metrics] Computing revenue by category…")
    category_totals = (
        joined_df
        .groupBy("category")
        .agg(
            _count("order_id").alias("order_count"),
            _sum("total_amount").alias("total_revenue")
        )
        .orderBy("category")
    ).toPandas().to_dict(orient="records")

    print("[generate_metrics] Computing customer age statistics…")
    age_stats = (
        joined_df
        .select("age")
        .agg(
            _avg("age").alias("average_age")
        )
    ).collect()[0].asDict()

    metrics = {
        "daily_totals": daily_totals,
        "category_totals": category_totals,
        "age_statistics": age_stats,
    }

    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "orders_metrics.json")
    print(f"[generate_metrics] Writing metrics to {metrics_file}")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    spark.stop()


def main() -> None:
    cwd = os.getcwd()
    default_orders = os.path.join(cwd, "data", "processing", "orders")
    default_customers = os.path.join(cwd, "data", "processing", "customers")
    default_products = os.path.join(cwd, "data", "processing", "products")
    default_metrics = os.path.join(cwd, "data", "curated", "metrics")

    orders_path = os.environ.get("PROCESSING_ORDERS_PATH", default_orders)
    customers_path = os.environ.get("PROCESSING_CUSTOMERS_PATH", default_customers)
    products_path = os.environ.get("PROCESSING_PRODUCTS_PATH", default_products)
    metrics_dir = os.environ.get("METRICS_DIR", default_metrics)

    run_metrics(orders_path, customers_path, products_path, metrics_dir)


if __name__ == "__main__":
    main()