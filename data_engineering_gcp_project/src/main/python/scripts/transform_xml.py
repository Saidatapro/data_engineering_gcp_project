"""
transform_xml.py
================

XML data often requires custom parsing since Spark does not ship with
native support for arbitrary XML schemas.  This script demonstrates how
to parse a simple XML file containing product records using Python's
built‑in ``xml.etree.ElementTree`` module, convert the parsed
structure into a Pandas DataFrame and then into a PySpark DataFrame for
further processing.  The resulting dataset is written to Parquet in
the processing zone.

The sample XML structure is defined as follows::

    <products>
      <product>
        <product_id>123</product_id>
        <product_name>Widget</product_name>
        <category>Tools</category>
        <price>19.99</price>
      </product>
      …
    </products>

New derived fields include a flag indicating whether a product is
considered 'premium' based on its price.  Adjust the threshold in
``is_premium`` as appropriate for your use case.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when


def parse_xml(file_path: str) -> List[Dict[str, str]]:
    """Parse an XML file and return a list of dictionaries.

    Parameters
    ----------
    file_path : str
        Path to the XML file.

    Returns
    -------
    List[Dict[str, str]]
        A list where each element corresponds to one <product> element
        in the XML.  All values are returned as strings.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    records: List[Dict[str, str]] = []
    for product in root.findall("product"):
        record = {
            "product_id": product.findtext("product_id"),
            "product_name": product.findtext("product_name"),
            "category": product.findtext("category"),
            "price": product.findtext("price"),
        }
        records.append(record)
    return records


def run_transform(raw_path: str, output_path: str) -> None:
    """Read product XML data, transform it and write as Parquet.

    Parameters
    ----------
    raw_path : str
        Location of the raw XML file.
    output_path : str
        Directory where Parquet output will be stored.
    """
    print(f"[transform_xml] Parsing XML from {raw_path}")
    records = parse_xml(raw_path)
    if not records:
        print("[transform_xml] No records found in XML file – nothing to do.")
        return

    # Convert to Pandas DataFrame first; this is convenient for small XML
    # documents.  For very large XML files you may want to stream and
    # convert chunks directly into Spark DataFrames.
    pandas_df = pd.DataFrame(records)
    # Ensure numeric types are correctly cast in the next step
    pandas_df["price"] = pandas_df["price"].astype(float)

    spark = (
        SparkSession.builder
        .appName("Transform XML Products")
        .getOrCreate()
    )

    products_df = spark.createDataFrame(pandas_df)

    print("[transform_xml] Adding derived columns…")
    products_df = (
        products_df
        .withColumn("price", col("price").cast("double"))
        .withColumn(
            "is_premium",
            when(col("price") >= 10.0, lit(True)).otherwise(lit(False)).cast("boolean")
        )
    )

    print("[transform_xml] Writing transformed data to Parquet…")
    (
        products_df
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    print(f"[transform_xml] Successfully wrote Parquet to {output_path}")

    spark.stop()


def main() -> None:
    raw_default = os.path.join(os.getcwd(), "data", "raw", "products.xml")
    processing_default = os.path.join(os.getcwd(), "data", "processing", "products")
    raw_path = os.environ.get("RAW_XML_PATH", raw_default)
    output_path = os.environ.get("PROCESSING_PRODUCTS_PATH", processing_default)
    run_transform(raw_path, output_path)


if __name__ == "__main__":
    main()