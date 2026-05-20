"""
Simple ETL pipeline example using pandas and scikit-learn.
This script is designed to run in Colab or any Python environment with pandas and scikit-learn installed.
"""

import importlib.util
import subprocess
import sys


def install_package(package_name: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def ensure_dependencies() -> None:
    for package_name in ["numpy", "pandas", "scikit-learn"]:
        if importlib.util.find_spec(package_name) is None:
            print(f"Installing missing dependency: {package_name}")
            install_package(package_name)


ensure_dependencies()

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_sample_data() -> pd.DataFrame:
    """Create a synthetic sales dataset for ETL demonstration."""
    data = {
        "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        "customer_id": [501, 502, 501, 503, 504, 502, 505, 506, 504, 507],
        "product": ["Widget", "Gadget", "Widget", "Widget", "Doohickey", "Gadget", "Widget", None, "Doohickey", "Gadget"],
        "category": ["A", "B", "A", "A", "C", "B", "A", "C", None, "B"],
        "quantity": [10, 5, 12, 7, 3, np.nan, 10, 8, 4, 6],
        "unit_price": [9.99, 19.99, 9.99, 9.99, 14.99, 19.99, np.nan, 14.99, 14.99, 19.99],
        "order_date": [
            "2024-01-12", "2024-01-15", "2024-02-02", "2024-02-05", "2024-02-10",
            "2024-03-01", "2024-03-08", "2024-03-12", "2024-03-15", "2024-04-01"
        ],
        "region": ["North", "West", "North", "East", "South", "West", "East", "South", "South", "West"]
    }
    df = pd.DataFrame(data)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and engineer features."""
    df = df.copy()

    # Convert order_date to datetime and extract date features.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month

    # Remove exact duplicate rows.
    df = df.drop_duplicates()

    # Fill missing categorical values with placeholder text.
    df["product"] = df["product"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")

    # Create a revenue feature.
    df["quantity"] = df["quantity"].astype(float)
    df["unit_price"] = df["unit_price"].astype(float)
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Basic imputation for numeric columns.
    df["quantity"] = df["quantity"].fillna(df["quantity"].median())
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
    df["revenue"] = df["revenue"].fillna(df["quantity"] * df["unit_price"])

    return df


def build_transformer() -> ColumnTransformer:
    """Create a transformer for numeric and categorical features."""
    numeric_features = ["quantity", "unit_price", "revenue", "order_year", "order_month"]
    categorical_features = ["product", "category", "region"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return transformer


def run_etl() -> None:
    """Execute ETL pipeline: load, preprocess, transform, save."""
    df = create_sample_data()
    print("Loaded data:\n", df)

    df_clean = preprocess_data(df)
    print("\nAfter preprocessing:\n", df_clean)

    transformer = build_transformer()
    X_transformed = transformer.fit_transform(df_clean)

    feature_names = (
        ["quantity", "unit_price", "revenue", "order_year", "order_month"] +
        list(transformer.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(["product", "category", "region"]))
    )

    df_final = pd.DataFrame(X_transformed, columns=feature_names)
    print("\nTransformed feature matrix shape:", df_final.shape)
    print(df_final.head())

    # Train/test split example
    train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42)
    print("\nTrain shape:", train_df.shape, "Test shape:", test_df.shape)

    train_df.to_csv("cleaned_sales_train.csv", index=False)
    test_df.to_csv("cleaned_sales_test.csv", index=False)
    print("\nSaved cleaned output files: cleaned_sales_train.csv, cleaned_sales_test.csv")


if __name__ == "__main__":
    run_etl()
