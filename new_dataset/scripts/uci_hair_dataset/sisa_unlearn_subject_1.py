import pandas as pd
import time
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"
models_folder = folder / "models"

FORGET_SUBJECT = 1

shard = pd.read_parquet(
    data_folder / "sisa_shard_1.parquet"
)

shard = shard[
    shard["subject"] != FORGET_SUBJECT
]

X = shard.drop(columns=["target", "subject"])
y = shard["target"]

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

start = time.time()

model.fit(X, y)

unlearning_time = time.time() - start

joblib.dump(
    model,
    models_folder / "sisa_shard_1_unlearned.joblib"
)

print("SISA UNLEARNING")
print("-" * 30)
print(f"Forgot subject    : {FORGET_SUBJECT}")
print(f"Remaining samples : {len(shard)}")
print(f"Unlearning time   : {unlearning_time:.4f} sec")
print("Model saved       : sisa_shard_1_unlearned.joblib")