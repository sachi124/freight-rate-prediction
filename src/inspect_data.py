# from pathlib import Path
# import pandas as pd

# DATA_DIR = path("data")

# for filename in ["train-test.csv", "validation.csv", "validation-predictions-template.csv"]:
#     Path = DATA_DIR / filename

#     if not path.exists():
#         print(f"Missing: {path}")
#         continue

#     df = pd.read_csv(path)

#     print(f"\n{filename}")
#     print("Shape:", df.shape)
#     print("Columns:", list(df.columns))
#     print("\nData Types:")
#     print(df.dtypes)
#     print("\nMissing Values:")
#     print(df.isnull().sum())
#     print("\nFirst Rows:")
#     print(df[:5])

import pandas as pd

train = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")

print("Train shape:", train.shape)
print("Validation shape:", validation.shape)
print("Train columns:", train.columns.tolist())
print("\nData types:")
print(train.dtypes)
print("\nMissing values:")
print(train.isna().sum())
print("\nDuplicate rows:", train.duplicated().sum())
print("\nUnique values:")
print(train.nunique())
print("\nNumeric summary:")
print(train.describe(include="all").T)