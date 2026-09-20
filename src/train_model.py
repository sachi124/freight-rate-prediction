from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

# General utilities

def find_target_column(frame: pd.DataFrame) -> str:
    """
    Detect the target column in train-test.csv.
    """

    possible_targets = [
        "posted_rate",
        "actual_rate",
        "load_rate",
        "freight_rate",
        "target",
        "price",
    ]

    for column in possible_targets:
        if column in frame.columns:
            return column

    raise ValueError(
        "Target column was not found. "
        f"Available columns: {list(frame.columns)}"
    )


def clean_target(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows with missing or invalid target values.
    """

    y_numeric = pd.to_numeric(y, errors="coerce")

    valid_rows = (
        y_numeric.notna()
        & np.isfinite(y_numeric)
    )

    X_clean = X.loc[valid_rows].copy()
    y_clean = y_numeric.loc[valid_rows].astype(float)

    return X_clean, y_clean

# Feature engineering
def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add model features and remove fields that should not be predictors.
    """

    result = frame.copy()

    # Identify likely date columns.
    date_columns = [
        column
        for column in result.columns
        if column.lower() in {
            "date",
            "load_date",
            "pickup_date",
            "ship_date",
        }
    ]

    for column in date_columns:
        dates = pd.to_datetime(
            result[column],
            errors="coerce",
        )

        result[f"{column}_year"] = dates.dt.year
        result[f"{column}_month"] = dates.dt.month
        result[f"{column}_day"] = dates.dt.day
        result[f"{column}_dayofweek"] = dates.dt.dayofweek
        result[f"{column}_dayofyear"] = dates.dt.dayofyear
        result[f"{column}_weekofyear"] = (
            dates.dt.isocalendar().week.astype("float64")
        )

        # Cyclic features preserve the relationship between
        # December and January and between Sunday and Monday.
        result[f"{column}_month_sin"] = np.sin(
            2 * np.pi * dates.dt.month / 12
        )
        result[f"{column}_month_cos"] = np.cos(
            2 * np.pi * dates.dt.month / 12
        )
        result[f"{column}_dow_sin"] = np.sin(
            2 * np.pi * dates.dt.dayofweek / 7
        )
        result[f"{column}_dow_cos"] = np.cos(
            2 * np.pi * dates.dt.dayofweek / 7
        )

        result = result.drop(columns=[column])

    # Freight-specific numerical feature.
    if {"distance", "weight"}.issubset(result.columns):
        distance = pd.to_numeric(
            result["distance"],
            errors="coerce",
        )
        weight = pd.to_numeric(
            result["weight"],
            errors="coerce",
        )

        result["weight_per_mile"] = (
            weight / distance.replace(0, np.nan)
        )

    # Combined route feature.
    if {"pickup", "delivery"}.issubset(result.columns):
        result["route"] = (
            result["pickup"].astype(str).str.strip()
            + " -> "
            + result["delivery"].astype(str).str.strip()
        )

    # The ID is for matching predictions, not for learning.
    if "load_id" in result.columns:
        result = result.drop(columns=["load_id"])

    return result

# Preprocessing and models
def make_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create the same preprocessing logic for every model.
    """

    numeric_columns = X.select_dtypes(
        include=["number", "bool"],
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"],
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False, 
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def make_model_pipelines(
    X: pd.DataFrame,
) -> dict[str, Pipeline]:
   
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),

        "extra_trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    pipelines = {}

    for name, estimator in models.items():
        pipelines[name] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(X),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

    return pipelines

# Train/holdout splitting
def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically when a date column is available.

    If there is no recognizable date column, preserve the current
    row order and use the first 80% for training.
    """

    date_column = next(
        (
            column
            for column in frame.columns
            if column.lower() in {
                "date",
                "load_date",
                "pickup_date",
                "ship_date",
            }
        ),
        None,
    )

    ordered = frame.copy()

    if date_column is not None:
        ordered["_temporary_sort_date"] = pd.to_datetime(
            ordered[date_column],
            errors="coerce",
        )

        ordered = ordered.sort_values(
            "_temporary_sort_date",
            na_position="last",
        )

        ordered = ordered.drop(
            columns=["_temporary_sort_date"],
        )

    split_index = int(len(ordered) * 0.80)

    development = ordered.iloc[:split_index].copy()
    holdout = ordered.iloc[split_index:].copy()

    return development, holdout

# Metrics and comparison
def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Calculate common regression metrics.
    """

    predictions = np.asarray(predictions, dtype=float)
    predictions = np.maximum(predictions, 0.01)

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    denominator = np.maximum(
        np.abs(y_true.to_numpy()),
        1e-8,
    )

    mape = np.mean(
        np.abs(
            (y_true.to_numpy() - predictions)
            / denominator
        )
    ) * 100

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "r2": float(r2),
    }


def compare_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """
    Fit and compare every model on exactly the same split.
    """

    results = []

    for name, pipeline in models.items():
        print(f"\nTraining: {name}")
        start_time = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        holdout_predictions = pipeline.predict(
            X_holdout,
        )

        metrics = calculate_metrics(
            y_holdout,
            holdout_predictions,
        )

        runtime_seconds = (
            time.perf_counter() - start_time
        )

        row = {
            "model": name,
            **metrics,
            "runtime_seconds": float(
                runtime_seconds
            ),
        }

        results.append(row)

        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(
            f"MAPE: {metrics['mape_percent']:.2f}%"
        )
        print(f"R2: {metrics['r2']:.4f}")
        print(
            f"Runtime: {runtime_seconds:.2f} seconds"
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="mae",
        ascending=True,
    ).reset_index(drop=True)

    best_model_name = str(
        comparison.loc[0, "model"]
    )

    return comparison, best_model_name

# Final prediction generation
def create_validation_predictions(
    final_model: Pipeline,
    validation_frame: pd.DataFrame,
    template_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions and preserve the template ID order.
    """

    if "load_id" not in validation_frame.columns:
        raise ValueError(
            "validation.csv must contain load_id."
        )

    validation_features = add_features(
        validation_frame.copy(),
    )

    raw_predictions = final_model.predict(
        validation_features,
    )

    raw_predictions = np.maximum(
        raw_predictions,
        0.01,
    )

    prediction_lookup = dict(
        zip(
            validation_frame["load_id"].astype(str),
            raw_predictions,
        )
    )

    if "load_id" not in template_frame.columns:
        raise ValueError(
            "validation-predictions-template.csv "
            "must contain load_id."
        )

    predictions = template_frame[["load_id"]].copy()
    predictions["load_id"] = predictions[
        "load_id"
    ].astype(str)

    predictions["predicted_rate"] = predictions[
        "load_id"
    ].map(prediction_lookup)

    if predictions["predicted_rate"].isna().any():
        missing_count = int(
            predictions["predicted_rate"].isna().sum()
        )

        raise ValueError(
            f"{missing_count} template IDs have no prediction."
        )

    predictions["predicted_rate"] = predictions[
        "predicted_rate"
    ].astype(float)

    return predictions

# Main workflow
def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = DATA_DIR / "train-test.csv"
    validation_path = DATA_DIR / "validation.csv"
    template_path = (
        DATA_DIR
        / "validation-predictions-template.csv"
    )

    print("Loading data...")

    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)
    template_frame = pd.read_csv(template_path)

    print(f"Development shape: {train_frame.shape}")
    print(f"Validation shape: {validation_frame.shape}")

    target = find_target_column(train_frame)
    print(f"Target column: {target}")

    # Chronological local validation
    development_frame, holdout_frame = (
        chronological_split(train_frame)
    )

    X_train_raw = development_frame.drop(
        columns=[target],
    )
    y_train_raw = development_frame[target]

    X_holdout_raw = holdout_frame.drop(
        columns=[target],
    )
    y_holdout_raw = holdout_frame[target]

    X_train_raw, y_train = clean_target(
        X_train_raw,
        y_train_raw,
    )

    X_holdout_raw, y_holdout = clean_target(
        X_holdout_raw,
        y_holdout_raw,
    )

    X_train = add_features(X_train_raw)
    X_holdout = add_features(X_holdout_raw)

    print(f"Training split rows: {len(X_train):,}")
    print(f"Holdout split rows: {len(X_holdout):,}")

    comparison_models = make_model_pipelines(
        X_train,
    )

    comparison, best_model_name = compare_models(
        comparison_models,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
    )

    comparison_path = (
        OUTPUT_DIR / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(
        f"\nSelected model: {best_model_name}"
    )

    # Final fit on all labeled development data
    X_all_raw = train_frame.drop(
        columns=[target],
    )
    y_all_raw = train_frame[target]

    X_all_raw, y_all = clean_target(
        X_all_raw,
        y_all_raw,
    )

    X_all = add_features(X_all_raw)

    final_models = make_model_pipelines(
        X_all,
    )

    final_model = final_models[
        best_model_name
    ]

    print(
        f"\nFitting {best_model_name} "
        "on all labeled data..."
    )

    final_model.fit(
        X_all,
        y_all,
    )

    # Predict validation.csv
    predictions = create_validation_predictions(
        final_model,
        validation_frame,
        template_frame,
    )

    prediction_path = (
        OUTPUT_DIR / "validation_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    # Save metadata for reproducibility
    best_result = comparison.iloc[0].to_dict()

    metadata = {
        "target_column": target,
        "selected_model": best_model_name,
        "development_rows": int(len(X_all)),
        "validation_rows": int(len(predictions)),
        "best_holdout_metrics": best_result,
        "random_state": RANDOM_STATE,
    }

    metadata_path = (
        OUTPUT_DIR / "run_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
    )

    print("\nFinished successfully.")
    print(f"Saved: {prediction_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {metadata_path}")
    print(
        f"Prediction rows: {len(predictions):,}"
    )
    print(
        f"Prediction columns: "
        f"{list(predictions.columns)}"
    )


if __name__ == "__main__":
    main()