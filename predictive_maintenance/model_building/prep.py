import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "predictive_maintenance/data/engine_data.csv"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
df = df.drop_duplicates().reset_index(drop=True)

features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]
target = "Engine Condition"

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

X = df[features]
y = df[target].astype(int)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print("Training shape:", Xtrain.shape)
print("Testing shape:", Xtest.shape)
print(ytrain.value_counts(normalize=True).round(4))
