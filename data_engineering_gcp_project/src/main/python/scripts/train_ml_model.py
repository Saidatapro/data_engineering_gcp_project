"""
train_ml_model.py
=================

This script builds a simple binary classification model using data
produced by the transformation jobs.  It reads pre‑processed
Parquet datasets for orders, customers and products from the
processing zone, performs a join to assemble a training set,
engineers features and trains a logistic regression model with
PySpark MLlib.  The trained model and the prediction results are
written to the curated zone.  Metrics such as AUC are printed to
stdout and can be captured by Airflow or Cloud Logging.

The objective of the model is to predict whether an order is
considered a "large" order based on its total amount, customer
attributes and product category.  A large order threshold can be
modified by setting the ``LARGE_ORDER_THRESHOLD`` environment
variable.
"""

import os
from typing import Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def load_datasets(spark: SparkSession, orders_path: str, customers_path: str, products_path: str):
    """Load the processed Parquet datasets and join them.

    Returns a Spark DataFrame containing the joined data.
    """
    orders_df = spark.read.parquet(orders_path)
    customers_df = spark.read.parquet(customers_path)
    products_df = spark.read.parquet(products_path)

    # Perform left joins to ensure that orders without a matching
    # customer or product still appear in the training set.  In a
    # production system you may want to drop such rows or fill
    # defaults.
    df = (
        orders_df
        .join(customers_df, on="customer_id", how="left")
        .join(products_df, on="product_id", how="left")
    )
    return df


def build_model(df, threshold: float) -> Tuple[Pipeline, any, float]:
    """Train a logistic regression model on the given DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame containing joined orders, customers and products.
    threshold : float
        Threshold above which an order is considered "large" and thus
        labelled as a positive instance.

    Returns
    -------
    Tuple[Pipeline, Transformer, float]
        The training pipeline, the fitted model and the AUC score on
        the test set.
    """
    # Label column: 1 if order total_amount >= threshold else 0
    df = df.withColumn(
        "label",
        when(col("total_amount") >= threshold, 1).otherwise(0)
    )

    # Fill null category values to avoid issues with StringIndexer
    df = df.fillna({"category": "Unknown"})

    # Define feature transformations
    category_indexer = StringIndexer(
        inputCol="category",
        outputCol="category_index",
        handleInvalid="keep"
    )
    # Assemble numeric and indexed categorical features
    feature_cols = [
        "quantity",
        "price",
        "age",
        "category_index",
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Define model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    # Build pipeline
    pipeline = Pipeline(stages=[category_indexer, assembler, lr])

    # Split data into training and testing sets
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    model = pipeline.fit(train_df)

    # Evaluate model using AUC
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    return pipeline, model, auc, predictions


def run_training(orders_path: str, customers_path: str, products_path: str, curated_dir: str) -> None:
    """Load data, train model and write outputs to the curated zone.

    Parameters
    ----------
    orders_path : str
        Path to processed orders Parquet directory.
    customers_path : str
        Path to processed customers Parquet directory.
    products_path : str
        Path to processed products Parquet directory.
    curated_dir : str
        Base directory where model and prediction Parquet files will be written.
    """
    threshold_env = os.environ.get("LARGE_ORDER_THRESHOLD", "50.0")
    try:
        threshold = float(threshold_env)
    except ValueError:
        print(f"[train_ml_model] Invalid threshold '{threshold_env}', defaulting to 50.0")
        threshold = 50.0

    spark = (
        SparkSession.builder
        .appName("Train ML Model")
        .getOrCreate()
    )

    print("[train_ml_model] Loading processed datasets…")
    df = load_datasets(spark, orders_path, customers_path, products_path)

    print("[train_ml_model] Training logistic regression model…")
    pipeline, model, auc, predictions = build_model(df, threshold)

    print(f"[train_ml_model] Model AUC on test set: {auc:.4f}")

    # Write predictions to curated zone
    predictions_out = os.path.join(curated_dir, "predictions")
    print(f"[train_ml_model] Writing predictions to {predictions_out}")
    (
        predictions
        .select(
            "order_id",
            "customer_id",
            "product_id",
            "total_amount",
            "prediction",
            "probability",
        )
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(predictions_out)
    )

    # Save the model
    model_path = os.path.join(curated_dir, "model")
    print(f"[train_ml_model] Saving model to {model_path}")
    # Remove existing model directory if present to avoid overwrite errors
    try:
        import shutil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
    except Exception as e:
        print(f"[train_ml_model] Warning: could not clean existing model path: {e}")
    model.write().overwrite().save(model_path)

    # Optionally persist metrics for external dashboards
    metrics_path = os.path.join(curated_dir, "metrics.json")
    print(f"[train_ml_model] Writing metrics to {metrics_path}")
    metrics_data = {
        "auc": auc,
        "large_order_threshold": threshold,
        "record_count": df.count(),
    }
    import json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    spark.stop()


def main() -> None:
    # Determine default paths relative to the project root
    cwd = os.getcwd()
    default_orders = os.path.join(cwd, "data", "processing", "orders")
    default_customers = os.path.join(cwd, "data", "processing", "customers")
    default_products = os.path.join(cwd, "data", "processing", "products")
    default_curated = os.path.join(cwd, "data", "curated")

    orders_path = os.environ.get("PROCESSING_ORDERS_PATH", default_orders)
    customers_path = os.environ.get("PROCESSING_CUSTOMERS_PATH", default_customers)
    products_path = os.environ.get("PROCESSING_PRODUCTS_PATH", default_products)
    curated_dir = os.environ.get("CURATED_DIR", default_curated)

    run_training(orders_path, customers_path, products_path, curated_dir)


if __name__ == "__main__":
    main()