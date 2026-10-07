import pandas as pd
from pathlib import Path

folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"

train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

FORGET_SUBJECT = 1

forget = train[train["subject"] == FORGET_SUBJECT]
retain = train[train["subject"] != FORGET_SUBJECT]

forget.to_parquet(
    data_folder / "sisa_forget_subject_1.parquet",
    index=False
)

retain.to_parquet(
    data_folder / "sisa_retain_subject_1.parquet",
    index=False
)

print("FORGET / RETAIN DATASET")
print("-" * 30)

print(f"Forget subject : {FORGET_SUBJECT}")
print(f"Forget samples : {len(forget)}")
print(f"Retain samples : {len(retain)}")

print()
print("Forget subjects:")
print(sorted(forget["subject"].unique()))

print()
print("Retain subjects:")
print(sorted(retain["subject"].unique()))