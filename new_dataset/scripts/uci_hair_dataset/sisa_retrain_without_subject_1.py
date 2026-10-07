import pandas as pd
import time
from pathlib import Path

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

FORGET_SUBJECT = 1


train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

forget = train[
    train["subject"] == FORGET_SUBJECT
]

retain = train[
    train["subject"] != FORGET_SUBJECT
]


print("SISA RETRAINING WITHOUT SUBJECT 1")
print("=" * 40)

print(f"Forget samples : {len(forget)}")
print(f"Retain samples : {len(retain)}")


models = []

total_time = 0


for i in range(1, 4):

    shard = pd.read_parquet(
        data_folder / f"sisa_shard_{i}.parquet"
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

    train_time = time.time() - start
    total_time += train_time

    models.append(model)

    print()
    print(f"Shard {i}")
    print(f"Samples       : {len(shard)}")
    print(f"Training time : {train_time:.4f} sec")


print()
print(f"Total training time : {total_time:.4f} sec")


# -------------------------------------------------
# EVALUATION DATA
# -------------------------------------------------

X_forget = forget.drop(
    columns=["target", "subject"]
)

y_forget = forget["target"]

X_retain = retain.drop(
    columns=["target", "subject"]
)

y_retain = retain["target"]


# -------------------------------------------------
# SISA PREDICTION
# -------------------------------------------------

def predict_sisa(models, X):

    probabilities = []

    for model in models:
        probabilities.append(
            model.predict_proba(X)
        )

    avg_probability = (
        sum(probabilities) / len(probabilities)
    )

    predictions = models[0].classes_[
        avg_probability.argmax(axis=1)
    ]

    return predictions


# -------------------------------------------------
# FORGET SET
# -------------------------------------------------

forget_predictions = predict_sisa(
    models,
    X_forget
)

forget_accuracy = accuracy_score(
    y_forget,
    forget_predictions
)

forget_precision = precision_score(
    y_forget,
    forget_predictions,
    average="macro",
    zero_division=0
)

forget_recall = recall_score(
    y_forget,
    forget_predictions,
    average="macro",
    zero_division=0
)

forget_f1 = f1_score(
    y_forget,
    forget_predictions,
    average="macro",
    zero_division=0
)


# -------------------------------------------------
# RETAIN SET
# -------------------------------------------------

retain_predictions = predict_sisa(
    models,
    X_retain
)

retain_accuracy = accuracy_score(
    y_retain,
    retain_predictions
)

retain_precision = precision_score(
    y_retain,
    retain_predictions,
    average="macro",
    zero_division=0
)

retain_recall = recall_score(
    y_retain,
    retain_predictions,
    average="macro",
    zero_division=0
)

retain_f1 = f1_score(
    y_retain,
    retain_predictions,
    average="macro",
    zero_division=0
)


print()
print("FORGET SET")
print("-" * 20)
print(f"Accuracy  : {forget_accuracy:.4f}")
print(f"Precision : {forget_precision:.4f}")
print(f"Recall    : {forget_recall:.4f}")
print(f"F1 Score  : {forget_f1:.4f}")


print()
print("RETAIN SET")
print("-" * 20)
print(f"Accuracy  : {retain_accuracy:.4f}")
print(f"Precision : {retain_precision:.4f}")
print(f"Recall    : {retain_recall:.4f}")
print(f"F1 Score  : {retain_f1:.4f}")


print()
print("=" * 40)
print("SISA RETRAINING COMPLETED")