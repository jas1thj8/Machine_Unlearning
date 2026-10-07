import pandas as pd
from pathlib import Path


folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"


train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

test = pd.read_parquet(
    data_folder / "uci_har_test.parquet"
)


print("Train shape:", train.shape)
print("Test shape :", test.shape)

print()
print("Missing values:")
print("Train:", train.isnull().sum().sum())
print("Test :", test.isnull().sum().sum())

print()
print("Duplicate rows:")
print("Train:", train.duplicated().sum())
print("Test :", test.duplicated().sum())

print()
print("Classes:")
print(sorted(train["target"].unique()))

print()
print("Training class distribution:")
print(train["target"].value_counts().sort_index())

print()
print("Subjects:")
print("Train:", sorted(train["subject"].unique()))
print("Test :", sorted(test["subject"].unique()))