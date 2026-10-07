import pandas as pd
from pathlib import Path


script_dir = Path(__file__).resolve().parent
folder = script_dir.parent
dataset = folder / "UCI HAR Dataset"


# Feature names
features = pd.read_csv(
    dataset / "features.txt",
    sep=r"\s+",
    header=None,
    names=["id", "name"]
)

feature_names = features["name"].tolist()


# Training data
X_train = pd.read_csv(
    dataset / "train" / "X_train.txt",
    sep=r"\s+",
    header=None
)

y_train = pd.read_csv(
    dataset / "train" / "y_train.txt",
    header=None
).iloc[:, 0]

subject_train = pd.read_csv(
    dataset / "train" / "subject_train.txt",
    header=None
).iloc[:, 0]


# Testing data
X_test = pd.read_csv(
    dataset / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    dataset / "test" / "y_test.txt",
    header=None
).iloc[:, 0]

subject_test = pd.read_csv(
    dataset / "test" / "subject_test.txt",
    header=None
).iloc[:, 0]


# Give features simple names
X_train.columns = [f"feature_{i+1:03d}" for i in range(X_train.shape[1])]
X_test.columns = X_train.columns


# Add target and subject
train = X_train.copy()
train["target"] = y_train
train["subject"] = subject_train

test = X_test.copy()
test["target"] = y_test
test["subject"] = subject_test


# Save
train.to_parquet(
    folder / "uci_har_train.parquet",
    index=False
)

test.to_parquet(
    folder / "uci_har_test.parquet",
    index=False
)


print()
print("UCI-HAR PREPROCESSING COMPLETED")
print("--------------------------------")
print(f"Training samples : {len(train)}")
print(f"Testing samples  : {len(test)}")
print(f"Features         : {len(feature_names)}")
print(f"Classes          : {train['target'].nunique()}")
print(f"Training subjects: {train['subject'].nunique()}")
print(f"Testing subjects : {test['subject'].nunique()}")
print()
print("Train shape:", train.shape)
print("Test shape :", test.shape)