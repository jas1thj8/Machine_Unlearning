from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"
models_folder = folder / "models"

train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

test = pd.read_parquet(
    data_folder / "uci_har_test.parquet"
)

X_test = test.drop(columns=["target", "subject"])
y_test = test["target"]

models = []

for i in range(1, 4):

    shard = pd.read_parquet(
        data_folder / f"sisa_shard_{i}.parquet"
    )

    X = shard.drop(columns=["target", "subject"])
    y = shard["target"]

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X, y)
    models.append(model)

probabilities = []

for model in models:
    probabilities.append(
        model.predict_proba(X_test)
    )

avg_probability = sum(probabilities) / len(probabilities)

predictions = models[0].classes_[
    avg_probability.argmax(axis=1)
]

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print("SISA LOGISTIC REGRESSION EVALUATION")
print("-" * 40)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")