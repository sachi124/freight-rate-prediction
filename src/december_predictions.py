from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42


def find_target_column(frame: pd.DataFrame) -> str:
    target_candidates = [
        "posted_rate",
        "actual_rate",
        "load_rate",
        "freight_rate",
        "target",
        "price",
    ]

    for column in target_candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "Target column was not found. "f"Available columns: {list(frame.columns)}"
    )


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    This must match the add_features() function in train_model.py.
    """

    result = frame.copy()

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

        result = result.drop(
            columns=[column],
        )

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

    if {"pickup", "delivery"}.issubset(result.columns):
        result["route"] = (
            result["pickup"].astype(str).str.strip()
            + " -> "
            + result["delivery"].astype(str).str.strip()
        )

    if "load_id" in result.columns:
        result = result.drop(
            columns=["load_id"],
        )

    return result


def make_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
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


def make_model(
    X: pd.DataFrame,
    model_name: str,
) -> Pipeline:
    """
    Use the same model configuration as the selected
    model from train_model.py.
    """

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models)}"
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(X),
            ),
            (
                "model",
                models[model_name],
            ),
        ]
    )


def build_december_input(
    train_frame: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.date_range(
        start="2025-12-01",
        end="2025-12-31",
        freq="D",
    )

    december = pd.DataFrame(
        {
            "pickup": ["Lexington"] * len(dates),
            "delivery": ["Fort Wayne"] * len(dates),
            "distance": [360.0] * len(dates),
            "equipment": ["Dry Van"] * len(dates),
            "weight": [32000.0] * len(dates),
            "date": dates,
        }
    )

    # Use historical values for the fixed Lexington-to-Fort Wayne route.
    route_rows = train_frame[
        (
            train_frame["pickup"].astype(str).str.strip()
            == "Lexington"
        )
        & (
            train_frame["delivery"].astype(str).str.strip()
            == "Fort Wayne"
        )
    ]

    extra_columns = [
        "pickup_lon",
        "delivery_lat",
        "market_index",
        "quote_signal",
        "pickup_lat",
        "delivery_lon",
    ]

    for column in extra_columns:
        if column not in train_frame.columns:
            continue

        if not route_rows.empty:
            value = route_rows[column].median()
        else:
            value = train_frame[column].median()

        december[column] = value

    return december


def validate_december_output(
    frame: pd.DataFrame,
) -> None:
    expected_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    if list(frame.columns) != expected_columns:
        raise ValueError(
            "Incorrect columns. "
            f"Expected: {expected_columns}"
        )

    if len(frame) != 31:
        raise ValueError(
            f"Expected 31 rows, received {len(frame)}."
        )

    dates = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    expected_dates = pd.date_range(
        "2025-12-01",
        "2025-12-31",
        freq="D",
    )

    if dates.isna().any():
        raise ValueError(
            "The December file contains invalid dates."
        )

    if set(dates) != set(expected_dates):
        raise ValueError(
            "Dates must cover every day from "
            "2025-12-01 to 2025-12-31."
        )

    if not frame["pickup"].eq("Lexington").all():
        raise ValueError(
            "Every pickup must be Lexington."
        )

    if not frame["delivery"].eq("Fort Wayne").all():
        raise ValueError(
            "Every delivery must be Fort Wayne."
        )

    if not np.isclose(
        frame["distance"],
        360.0,
    ).all():
        raise ValueError(
            "Every distance must be 360.0."
        )

    if not frame["equipment"].eq("Dry Van").all():
        raise ValueError(
            "Every equipment value must be Dry Van."
        )

    if not np.isclose(
        frame["weight"],
        32000.0,
    ).all():
        raise ValueError(
            "Every weight must be 32000.0."
        )

    rates = pd.to_numeric(
        frame["predicted_rate"],
        errors="coerce",
    )

    if rates.isna().any():
        raise ValueError(
            "predicted_rate contains invalid values."
        )

    if not np.isfinite(rates).all():
        raise ValueError(
            "predicted_rate contains non-finite values."
        )

    if (rates <= 0).any():
        raise ValueError(
            "predicted_rate must be positive."
        )


def main() -> None:
    train_path = DATA_DIR / "train-test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    train_frame = pd.read_csv(train_path)
    target = find_target_column(train_frame)

    X_raw = train_frame.drop(
        columns=[target],
    )
    y = pd.to_numeric(
        train_frame[target],
        errors="coerce",
    )

    valid_rows = (
        y.notna()
        & np.isfinite(y)
    )

    X_raw = X_raw.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(float)

    X_train = add_features(X_raw)

    # Use the model that won model_comparison.csv.
    # Change this value to the actual winner.
    selected_model_name = "hist_gradient_boosting"

    model = make_model(
        X_train,
        selected_model_name,
    )

    print(
        f"Training {selected_model_name} "
        "on all labeled data..."
    )

    model.fit(
        X_train,
        y,
    )

    december_input = build_december_input(train_frame,)
    december_features = add_features(
        december_input.copy(),
    )

    predictions = model.predict(
        december_features,
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    predictions = np.maximum(
        predictions,
        0.01,
    )

    december_output = december_input.copy()
    december_output["predicted_rate"] = predictions

    december_output = december_output[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ]

    validate_december_output(
        december_output,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "december_chart_inputs.csv"
    )

    december_output.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"Saved {len(december_output)} rows to "
        f"{output_path}"
    )

    print(december_output.head())


if __name__ == "__main__":
    main()