import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "data.csv"

N_SHARDS = 5
FORGET_RATIO = 0.10

RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Y:\\VSc\\MUL\\breast-cancer.csv")

print("Original dataset:", df.shape)


# ============================================================
# 2. CLEAN DATASET
# ============================================================

# Remove unnecessary columns
columns_to_remove = []

for col in ["id", "ID", "Unnamed: 32"]:
    if col in df.columns:
        columns_to_remove.append(col)

df = df.drop(columns=columns_to_remove)


# Target conversion
if "diagnosis" in df.columns:

    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

else:
    raise ValueError(
        "Target column 'diagnosis' was not found."
    )


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# Reset indexes
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# ============================================================
# 4. CREATE SHARDS
# ============================================================

def create_shards(X, y, n_shards):

    indices = np.arange(len(X))

    rng = np.random.default_rng(RANDOM_STATE)

    rng.shuffle(indices)

    shard_indices = np.array_split(indices, n_shards)

    shards = []

    for i, indices in enumerate(shard_indices):

        X_shard = X.iloc[indices].copy()
        y_shard = y.iloc[indices].copy()

        shards.append({
            "X": X_shard,
            "y": y_shard,
            "indices": indices
        })

        print(
            f"Shard {i}: "
            f"{len(indices)} samples"
        )

    return shards


shards = create_shards(
    X_train,
    y_train,
    N_SHARDS
)


# ============================================================
# 5. MODEL CREATION
# ============================================================

def create_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ])


# ============================================================
# 6. TRAIN EACH SHARD
# ============================================================

def train_shards(shards):
    """Train one independent logistic-regression model per SISA shard."""
    models = []
    training_time = 0.0

    for shard_id, shard in enumerate(shards):
        model = create_model()
        start_time = time.time()
        model.fit(shard["X"], shard["y"])
        elapsed_time = time.time() - start_time
        training_time += elapsed_time
        models.append(model)
        print(
            f"Shard {shard_id} trained on {len(shard['y'])} samples "
            f"in {elapsed_time:.4f} seconds"
        )

    print(f"\nTotal SISA training time: {training_time:.4f} seconds")
    return models


models = train_shards(shards)


# ============================================================
# 7. AGGREGATE PREDICTIONS
# ============================================================

def predict_sisa(models, X):

    predictions = []

    probabilities = []

    for model in models:

        pred = model.predict(X)

        prob = model.predict_proba(X)[:, 1]

        predictions.append(pred)

        probabilities.append(prob)

    predictions = np.array(predictions)

    probabilities = np.array(probabilities)

    # Average probabilities from all shard models
    average_probability = probabilities.mean(axis=0)

    final_prediction = (
        average_probability >= 0.5
    ).astype(int)

    return final_prediction


# ============================================================
# 8. EVALUATE ORIGINAL SISA MODEL
# ============================================================

original_predictions = predict_sisa(
    models,
    X_test
)

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

original_f1 = f1_score(
    y_test,
    original_predictions
)

print("\n" + "=" * 60)
print("ORIGINAL SISA MODEL")
print("=" * 60)

print(
    f"Accuracy: {original_accuracy:.4f}"
)

print(
    f"F1 Score: {original_f1:.4f}"
)


# ============================================================
# 9. CREATE FORGET SET
# ============================================================

total_forget = int(
    len(X_train) * FORGET_RATIO
)

rng = np.random.default_rng(
    RANDOM_STATE
)

forget_indices = rng.choice(
    len(X_train),
    size=total_forget,
    replace=False
)

forget_indices = set(
    forget_indices.tolist()
)

print("\n" + "=" * 60)
print("FORGET SET")
print("=" * 60)

print(
    "Forget samples:",
    len(forget_indices)
)


# ============================================================
# 10. FIND AFFECTED SHARDS
# ============================================================

affected_shards = []

for shard_id, shard in enumerate(shards):

    shard_original_indices = set(
        shard["indices"].tolist()
    )

    intersection = (
        shard_original_indices
        & forget_indices
    )

    if intersection:

        affected_shards.append(
            shard_id
        )

        print(
            f"Shard {shard_id}: "
            f"{len(intersection)} samples to forget"
        )


# ============================================================
# 11. REMOVE FORGET SAMPLES
# ============================================================

updated_shards = []

for shard_id, shard in enumerate(shards):

    X_shard = shard["X"].copy()
    y_shard = shard["y"].copy()

    keep_positions = []

    for position, original_index in enumerate(
        shard["indices"]
    ):

        if original_index not in forget_indices:

            keep_positions.append(position)

    X_remaining = X_shard.iloc[
        keep_positions
    ].copy()

    y_remaining = y_shard.iloc[
        keep_positions
    ].copy()

    updated_shards.append({
        "X": X_remaining,
        "y": y_remaining,
        "indices": shard["indices"][
            keep_positions
        ]
    })


# ============================================================
# 12. RETRAIN ONLY AFFECTED SHARDS
# ============================================================

start_time = time.time()

updated_models = models.copy()

for shard_id in affected_shards:

    print(
        f"\nRetraining affected shard "
        f"{shard_id}..."
    )

    shard = updated_shards[shard_id]

    model = create_model()

    model.fit(
        shard["X"],
        shard["y"]
    )

    updated_models[shard_id] = model


unlearning_time = time.time() - start_time

print(
    f"\nSISA unlearning time: "
    f"{unlearning_time:.4f} seconds"
)


# ============================================================
# 13. EVALUATE UNLEARNED MODEL
# ============================================================

unlearned_predictions = predict_sisa(
    updated_models,
    X_test
)

unlearned_accuracy = accuracy_score(
    y_test,
    unlearned_predictions
)

unlearned_f1 = f1_score(
    y_test,
    unlearned_predictions
)

print("\n" + "=" * 60)
print("UNLEARNED SISA MODEL")
print("=" * 60)

print(
    f"Accuracy: {unlearned_accuracy:.4f}"
)

print(
    f"F1 Score: {unlearned_f1:.4f}"
)


# ============================================================
# 14. FULL RETRAINING BASELINE
# ============================================================

retain_mask = np.ones(
    len(X_train),
    dtype=bool
)

retain_mask[
    list(forget_indices)
] = False

X_retain = X_train.iloc[
    retain_mask
]

y_retain = y_train.iloc[
    retain_mask
]


start_time = time.time()

full_retrain_model = create_model()

full_retrain_model.fit(
    X_retain,
    y_retain
)

full_retrain_time = (
    time.time() - start_time
)


retrain_predictions = (
    full_retrain_model.predict(X_test)
)

retrain_accuracy = accuracy_score(
    y_test,
    retrain_predictions
)

retrain_f1 = f1_score(
    y_test,
    retrain_predictions
)


# ============================================================
# 15. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"\nOriginal SISA:"
)

print(
    f"Accuracy = "
    f"{original_accuracy:.4f}"
)

print(
    f"F1       = "
    f"{original_f1:.4f}"
)


print(
    f"\nSISA Unlearned:"
)

print(
    f"Accuracy = "
    f"{unlearned_accuracy:.4f}"
)

print(
    f"F1       = "
    f"{unlearned_f1:.4f}"
)

print(
    f"Unlearning time = "
    f"{unlearning_time:.4f} sec"
)


print(
    f"\nFull Retraining:"
)

print(
    f"Accuracy = "
    f"{retrain_accuracy:.4f}"
)

print(
    f"F1       = "
    f"{retrain_f1:.4f}"
)

print(
    f"Retraining time = "
    f"{full_retrain_time:.4f} sec"
)


# ============================================================
# 16. TIME SAVING
# ============================================================

if full_retrain_time > 0:

    speedup = (
        full_retrain_time /
        unlearning_time
    )

    print(
        f"\nSISA speedup: "
        f"{speedup:.2f}x"
    )
