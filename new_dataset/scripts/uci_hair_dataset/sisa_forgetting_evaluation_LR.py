import pandas as pd
import joblib
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


X_forget = forget.drop(columns=["target", "subject"])
y_forget = forget["target"]

X_retain = retain.drop(columns=["target", "subject"])
y_retain = retain["target"]


def evaluate_model(models, X, y):

    probabilities = []

    for model in models:
        probabilities.append(
            model.predict_proba(X)
        )

    avg_probability = sum(probabilities) / len(probabilities)

    predictions = models[0].classes_[
        avg_probability.argmax(axis=1)
    ]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, precision, recall, f1


def print_results(name, forget_results, retain_results):

    print()
    print(name)
    print("-" * len(name))

    print("Forget Set:")
    print(f"Accuracy  : {forget_results[0]:.4f}")
    print(f"Precision : {forget_results[1]:.4f}")
    print(f"Recall    : {forget_results[2]:.4f}")
    print(f"F1 Score  : {forget_results[3]:.4f}")

    print()
    print("Retain Set:")
    print(f"Accuracy  : {retain_results[0]:.4f}")
    print(f"Precision : {retain_results[1]:.4f}")
    print(f"Recall    : {retain_results[2]:.4f}")
    print(f"F1 Score  : {retain_results[3]:.4f}")


print("SISA FORGETTING EVALUATION")
print("=" * 40)

print(f"Forget Subject : {FORGET_SUBJECT}")
print(f"Forget Samples : {len(forget)}")
print(f"Retain Samples : {len(retain)}")


# -------------------------------------------------
# 1. ORIGINAL SISA
# -------------------------------------------------

original_models = []

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

    original_models.append(model)


original_forget = evaluate_model(
    original_models,
    X_forget,
    y_forget
)

original_retain = evaluate_model(
    original_models,
    X_retain,
    y_retain
)


print_results(
    "Original SISA",
    original_forget,
    original_retain
)


# -------------------------------------------------
# 2. SISA AFTER UNLEARNING
# -------------------------------------------------

unlearned_models = []

model_1 = joblib.load(
    models_folder / "sisa_shard_1_unlearned.joblib"
)

unlearned_models.append(model_1)


for i in [2, 3]:

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

    unlearned_models.append(model)


unlearned_forget = evaluate_model(
    unlearned_models,
    X_forget,
    y_forget
)

unlearned_retain = evaluate_model(
    unlearned_models,
    X_retain,
    y_retain
)


print_results(
    "SISA After Unlearning",
    unlearned_forget,
    unlearned_retain
)


# -------------------------------------------------
# 3. FULL RETRAINING
# -------------------------------------------------

full_retrain_model = joblib.load(
    models_folder / "full_retrain_subject_1.joblib"
)


full_retrain_forget = (
    accuracy_score(
        y_forget,
        full_retrain_model.predict(X_forget)
    ),
    precision_score(
        y_forget,
        full_retrain_model.predict(X_forget),
        average="macro",
        zero_division=0
    ),
    recall_score(
        y_forget,
        full_retrain_model.predict(X_forget),
        average="macro",
        zero_division=0
    ),
    f1_score(
        y_forget,
        full_retrain_model.predict(X_forget),
        average="macro",
        zero_division=0
    )
)


full_retrain_retain = (
    accuracy_score(
        y_retain,
        full_retrain_model.predict(X_retain)
    ),
    precision_score(
        y_retain,
        full_retrain_model.predict(X_retain),
        average="macro",
        zero_division=0
    ),
    recall_score(
        y_retain,
        full_retrain_model.predict(X_retain),
        average="macro",
        zero_division=0
    ),
    f1_score(
        y_retain,
        full_retrain_model.predict(X_retain),
        average="macro",
        zero_division=0
    )
)


print_results(
    "Full Retraining",
    full_retrain_forget,
    full_retrain_retain
)


print()
print("=" * 40)
print("EVALUATION COMPLETED")