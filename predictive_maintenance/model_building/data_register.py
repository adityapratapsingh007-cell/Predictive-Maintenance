import os
import pandas as pd

RAW_PATH = "predictive_maintenance/data/engine_data.csv"

required_columns = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered and validated successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Target distribution:")
print(df["Engine Condition"].value_counts())
print("Target proportion:")
print(df["Engine Condition"].value_counts(normalize=True).round(4))
