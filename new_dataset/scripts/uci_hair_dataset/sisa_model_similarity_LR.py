import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression


folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"
models_folder = folder / "models"


FORGET_SUBJECT = 1


train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

test = pd.read_parquet(
    data_folder / "uci_har_test.parquet"
)


forget = train[
    train["subject"] == FORGET_SUBJECT
]

retain = train[
    train["subject"] != FORGET_SUBJECT
]


X_forget = forget.drop(
    columns=["target", "subject"]
)

X_retain = retain.drop(
    columns=["target", "subject"]
)

X_test = test.drop(
    columns=["target", "subject"]
)


# -------------------------------------------------
# SISA AFTER UNLEARNING
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

    X = shard.drop(
        columns=["target", "subject"]
    )

    y = shard["target"]

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X, y)

    unlearned_models.append(model)


# -------------------------------------------------
# SISA RETRAINED WITHOUT SUBJECT 1
# -------------------------------------------------

retrained_models = []

for i in range(1, 4):

    shard = pd.read_parquet(
        data_folder / f"sisa_shard_{i}.parquet"
    )

    shard = shard[
        shard["subject"] != FORGET_SUBJECT
    ]

    X = shard.drop(
        columns=["target", "subject"]
    )

    y = shard["target"]

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X, y)

    retrained_models.append(model)


# -------------------------------------------------
# SISA PREDICTIONS
# -------------------------------------------------

def get_predictions(models, X):

    probabilities = []

    for model in models:
        probabilities.append(
            model.predict_proba(X)
        )

    probabilities = np.array(probabilities)

    average_probability = probabilities.mean(
        axis=0
    )

    predictions = models[0].classes_[
        average_probability.argmax(axis=1)
    ]

    return predictions, average_probability


# -------------------------------------------------
# FORGET SET
# -------------------------------------------------

unlearned_forget_pred, unlearned_forget_prob = (
    get_predictions(
        unlearned_models,
        X_forget
    )
)

retrained_forget_pred, retrained_forget_prob = (
    get_predictions(
        retrained_models,
        X_forget
    )
)


# -------------------------------------------------
# RETAIN SET
# -------------------------------------------------

unlearned_retain_pred, unlearned_retain_prob = (
    get_predictions(
        unlearned_models,
        X_retain
    )
)

retrained_retain_pred, retrained_retain_prob = (
    get_predictions(
        retrained_models,
        X_retain
    )
)


# -------------------------------------------------
# TEST SET
# -------------------------------------------------

unlearned_test_pred, unlearned_test_prob = (
    get_predictions(
        unlearned_models,
        X_test
    )
)

retrained_test_pred, retrained_test_prob = (
    get_predictions(
        retrained_models,
        X_test
    )
)


# -------------------------------------------------
# PREDICTION AGREEMENT
# -------------------------------------------------

forget_agreement = np.mean(
    unlearned_forget_pred ==
    retrained_forget_pred
)

retain_agreement = np.mean(
    unlearned_retain_pred ==
    retrained_retain_pred
)

test_agreement = np.mean(
    unlearned_test_pred ==
    retrained_test_pred
)


# -------------------------------------------------
# PROBABILITY DIFFERENCE
# -------------------------------------------------

forget_probability_difference = np.mean(
    np.abs(
        unlearned_forget_prob -
        retrained_forget_prob
    )
)

retain_probability_difference = np.mean(
    np.abs(
        unlearned_retain_prob -
        retrained_retain_prob
    )
)

test_probability_difference = np.mean(
    np.abs(
        unlearned_test_prob -
        retrained_test_prob
    )
)


# -------------------------------------------------
# COEFFICIENT DIFFERENCE
# -------------------------------------------------

coefficient_differences = []

for unlearned_model, retrained_model in zip(
    unlearned_models,
    retrained_models
):

    difference = np.mean(
        np.abs(
            unlearned_model.coef_ -
            retrained_model.coef_
        )
    )

    coefficient_differences.append(
        difference
    )


# -------------------------------------------------
# RESULTS
# -------------------------------------------------

print()
print("SISA MODEL SIMILARITY")
print("=" * 40)

print()
print("Prediction Agreement")
print("--------------------")
print(
    f"Forget Set : "
    f"{forget_agreement:.6f}"
)

print(
    f"Retain Set : "
    f"{retain_agreement:.6f}"
)

print(
    f"Test Set   : "
    f"{test_agreement:.6f}"
)


print()
print("Mean Probability Difference")
print("---------------------------")

print(
    f"Forget Set : "
    f"{forget_probability_difference:.8f}"
)

print(
    f"Retain Set : "
    f"{retain_probability_difference:.8f}"
)

print(
    f"Test Set   : "
    f"{test_probability_difference:.8f}"
)


print()
print("Mean Coefficient Difference")
print("---------------------------")

for i, difference in enumerate(
    coefficient_differences,
    start=1
):

    print(
        f"Shard {i}: "
        f"{difference:.8f}"
    )


print()
print("=" * 40)
print("MODEL SIMILARITY EVALUATION COMPLETED")