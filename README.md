# Freight Rate Prediction

Machine-learning pipeline for predicting freight rates for unseen shipment loads.

## Project objective

The objective is to train a regression model using labeled development data and generate predicted freight rates for every load in the validation dataset.

The project uses:

- `data/train_test.csv` as labeled development data.
- `data/validation.csv` as the unlabeled prediction data.
- `data/validation_predictions_template.csv` as the required prediction template.

The target variable in the development data is:

```text
posted_rate
```

## Project structure

```text
Frieght-Rate-ML/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── src/
│   ├── train_model.py
│   └── december_predictions.py
├── outputs/
│   ├── validation_predictions.csv
│   ├── december_chart_inputs.csv
│   ├── model_comparison.csv
│   ├── run_metadata.json
│   └── scorer_results/
│       └── candidate_december.png
├── score.py
├── requirements.txt
├── freight_rate_assessment_report.pdf
├── README.md
└── .gitignore
```

## Data overview

The development dataset contains:

- 48,000 rows.
- 14 columns.
- `posted_rate` as the target variable.

The validation dataset contains:

- 12,000 rows.
- 13 columns.
- A unique `load_id` for each load.

The features include:

- Pickup and delivery locations.
- Pickup and delivery latitude and longitude.
- Distance.
- Equipment type.
- Weight.
- Date.
- Market index.
- Quote signal.

The `load_id` column is used to match predictions to the required output template and is excluded from model training.

## Data preparation

The pipeline performs the following data preparation steps:

- Converts the target column to numeric values.
- Removes rows with missing or invalid target values.
- Imputes missing numeric values with the median.
- Imputes missing categorical values with the most frequent category.
- One-hot encodes categorical features.
- Ignores unknown categories during validation and prediction.
- Removes `load_id` from model features.
- Protects the weight-per-mile calculation against zero distance values.
- Ensures final predicted rates are positive.

## Feature engineering

Date features are extracted from the original date column:

- Year.
- Month.
- Day.
- Day of week.
- Day of year.
- Week of year.
- Cyclic month features.
- Cyclic day-of-week features.

Additional features include:

- Combined route from pickup and delivery.
- Weight per mile.

The route feature is created as:

```text
pickup -> delivery
```

For example:

```text
Lexington -> Fort Wayne
```

## Training and validation approach

A chronological 80/20 split was used for local model validation.

The labeled data was sorted by date:

- The earliest 80 percent, consisting of 38,400 rows, was used for model development.
- The latest 20 percent, consisting of 9,600 rows, was used as a holdout set.

A chronological split was selected because freight rates can vary over time due to demand, capacity, seasonality, and market conditions. This approach better represents prediction on future-like observations than randomly mixing earlier and later rows.

After model comparison, the selected model was retrained using all 48,000 valid labeled rows before generating predictions for the 12,000 validation loads.

## Models compared

The following regression models were compared:

- `RandomForestRegressor`
- `HistGradientBoostingRegressor`
- `ExtraTreesRegressor`

All models used the same:

- Feature engineering.
- Preprocessing.
- Training split.
- Holdout split.
- Evaluation metrics.

The primary selection metric was mean absolute error, or MAE. RMSE, MAPE, R², and runtime were also recorded.

## Model comparison results

| Model | MAE | RMSE | MAPE | R² | Runtime |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 128.9614 | 634.2302 | 6.25% | 0.8270 | 70.09 sec |
| Random Forest | 154.2157 | 662.9257 | 6.85% | 0.8110 | 688.41 sec |
| Extra Trees | 158.0400 | 691.9174 | 6.93% | 0.7941 | 1,565.66 sec |

## Selected model

`HistGradientBoostingRegressor` was selected because it produced the best holdout results:

- Lowest MAE.
- Lowest RMSE.
- Lowest MAPE.
- Highest R².
- Fastest runtime among the tested models.

The selected model was refit on all valid labeled development rows before final inference.

## December prediction scenario

A separate script generates predictions for the required fixed December scenario.

| Field | Value |
|---|---|
| Pickup | Lexington |
| Delivery | Fort Wayne |
| Distance | 360 miles |
| Equipment | Dry Van |
| Weight | 32,000 lb |
| Dates | 2025-12-01 through 2025-12-31 |

Only the date changes between the 31 rows. The selected model and the same feature-engineering logic are used to generate the daily predictions.

The final December file contains exactly these columns:

```text
pickup,delivery,distance,equipment,weight,date,predicted_rate
```

## Installation

Create and activate a Python virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Requirements

The main dependencies are:

```text
matplotlib
numpy
pandas
scikit-learn
```

## Run the project

Run the main training and prediction pipeline:

```powershell
python src\train_model.py
```

This generates:

```text
outputs\validation_predictions.csv
outputs\model_comparison.csv
outputs\run_metadata.json
```

Generate the fixed December predictions:

```powershell
python src\december_predictions.py
```

This generates:

```text
outputs\december_chart_inputs.csv
```

## Run the official scorer

Run the supplied scorer from the project root:

```powershell
python score.py --predictions outputs\validation_predictions.csv --december-predictions outputs\december_chart_inputs.csv --output-dir outputs\scorer_results
```

Successful validation output:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: outputs\scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

## Final outputs

The main final files are:

```text
outputs\validation_predictions.csv
outputs\december_chart_inputs.csv
outputs\scorer_results\candidate_december.png
```

The validation prediction file contains:

```text
load_id,predicted_rate
```

The file contains exactly 12,000 prediction rows.

The December file contains exactly 31 rows and the required fixed inputs.

## Report

The project also includes:

```text
freight_rate_assessment_report.pdf
```

The report documents:

- Data exploration.
- Data-quality handling.
- Feature engineering.
- Train/holdout splitting.
- Model comparison.
- Model selection.
- Code walkthrough.
- December prediction process.
- Official scorer validation.
- Fixed December prediction chart.

## Reproducibility

The complete workflow is:

```powershell
python src\train_model.py
python src\december_predictions.py
python score.py --predictions outputs\validation_predictions.csv --december-predictions outputs\december_chart_inputs.csv --output-dir outputs\scorer_results
```

The virtual environment, Python cache files, local editor settings, and secret files are excluded using `.gitignore`.
