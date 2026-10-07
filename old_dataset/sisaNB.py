import time
import random
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
CSV_PATH = r"Y:\VSc\MUL\breast-cancer.csv"

TARGET_COLUMN = None

NUM_SHARDS = 3

FORGET_SHARD = 0
FORGET_POSITION = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("SISA + GAUSSIAN NAIVE BAYES")
print("=" * 70)

print("Dataset shape:", df.shape)


# ============================================================
# TARGET DETECTION
# ============================================================

if TARGET_COLUMN is None:

    candidates = [
        "target",
        "label",
        "class",
        "failure",
        "battery_failure",
        "failure_label",
        "status",
        "diagnosis",
        "y"
    ]

    for column in df.columns:

        if column.lower().strip() in candidates:

            TARGET_COLUMN = column
            break


if TARGET_COLUMN is None:

    raise ValueError(
        "Target column not found. "
        "Set TARGET_COLUMN manually."
    )


print("Target:", TARGET_COLUMN)


# ============================================================
# REMOVE ID COLUMNS
# ============================================================

drop_columns = []

for column in df.columns:

    if column.lower().strip() in [
        "id",
        "unnamed: 0",
        "unnamed: 32"
    ]:

        drop_columns.append(column)


if drop_columns:

    df = df.drop(
        columns=drop_columns
    )


# ============================================================
# FEATURES / TARGET
# ============================================================

y_raw = df[TARGET_COLUMN]

X_df = df.drop(
    columns=[TARGET_COLUMN]
).copy()


# ============================================================
# FEATURE PROCESSING
# ============================================================

for column in X_df.columns:

    numeric = pd.to_numeric(
        X_df[column],
        errors="coerce"
    )

    if numeric.notna().all():

        X_df[column] = numeric

    else:

        encoder = LabelEncoder()

        X_df[column] = encoder.fit_transform(
            X_df[column].astype(str)
        )


# ============================================================
# TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y_raw.astype(str)
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid = X_df.notna().all(axis=1)

X_df = X_df.loc[valid].reset_index(
    drop=True
)

y = y[
    valid.to_numpy()
]


X = X_df.to_numpy(
    dtype=np.float32
)

y = y.astype(
    np.int64
)


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# SISA SHARDING
# ============================================================

def create_shards(
    y_data,
    num_shards
):

    rng = np.random.default_rng(
        SEED
    )

    shards = [
        []
        for _ in range(num_shards)
    ]


    for cls in np.unique(
        y_data
    ):

        indices = np.where(
            y_data == cls
        )[0]

        rng.shuffle(
            indices
        )

        splits = np.array_split(
            indices,
            num_shards
        )


        for i in range(
            num_shards
        ):

            shards[i].extend(
                splits[i].tolist()
            )


    for shard in shards:

        rng.shuffle(
            shard
        )


    return shards


shard_indices = create_shards(
    y_train,
    NUM_SHARDS
)


# ============================================================
# MODEL
# ============================================================

import time
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
CSV_PATH = r"Y:\VSc\MUL\breast-cancer.csv"

TARGET_COLUMN = None

NUM_SHARDS = 3

FORGET_SHARD = 0
FORGET_POSITION = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("SISA + GAUSSIAN NAIVE BAYES")
print("=" * 70)

print("Dataset shape:", df.shape)


# ============================================================
# TARGET DETECTION
# ============================================================

if TARGET_COLUMN is None:

    candidates = [
        "target",
        "label",
        "class",
        "failure",
        "battery_failure",
        "failure_label",
        "status",
        "diagnosis",
        "y"
    ]

    for column in df.columns:

        if column.lower().strip() in candidates:

            TARGET_COLUMN = column
            break


if TARGET_COLUMN is None:

    raise ValueError(
        "Target column not found. "
        "Set TARGET_COLUMN manually."
    )


print("Target:", TARGET_COLUMN)


# ============================================================
# REMOVE ID COLUMNS
# ============================================================

drop_columns = []

for column in df.columns:

    if column.lower().strip() in [
        "id",
        "unnamed: 0",
        "unnamed: 32"
    ]:

        drop_columns.append(column)


if drop_columns:

    df = df.drop(
        columns=drop_columns
    )


# ============================================================
# FEATURES / TARGET
# ============================================================

y_raw = df[TARGET_COLUMN]

X_df = df.drop(
    columns=[TARGET_COLUMN]
).copy()


# ============================================================
# FEATURE PROCESSING
# ============================================================

for column in X_df.columns:

    numeric = pd.to_numeric(
        X_df[column],
        errors="coerce"
    )

    if numeric.notna().all():

        X_df[column] = numeric

    else:

        encoder = LabelEncoder()

        X_df[column] = encoder.fit_transform(
            X_df[column].astype(str)
        )


# ============================================================
# TARGET ENCODING
# ============================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y_raw.astype(str)
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid = X_df.notna().all(axis=1)

X_df = X_df.loc[valid].reset_index(
    drop=True
)

y = y[
    valid.to_numpy()
]


X = X_df.to_numpy(
    dtype=np.float32
)

y = y.astype(
    np.int64
)


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# SISA SHARDING
# ============================================================

def create_shards(
    y_data,
    num_shards
):

    rng = np.random.default_rng(
        SEED
    )

    shards = [
        []
        for _ in range(num_shards)
    ]


    for cls in np.unique(
        y_data
    ):

        indices = np.where(
            y_data == cls
        )[0]

        rng.shuffle(
            indices
        )

        splits = np.array_split(
            indices,
            num_shards
        )


        for i in range(
            num_shards
        ):

            shards[i].extend(
                splits[i].tolist()
            )


    for shard in shards:

        rng.shuffle(
            shard
        )


    return shards


shard_indices = create_shards(
    y_train,
    NUM_SHARDS
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    return GaussianNB() 
# ============================================================
# PROBABILITY
# ============================================================

def probabilities(
    model,
    X
):

    return model.predict_proba(X)


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(
    models,
    X,
    y
):

    all_probs = []

    for model in models:

        all_probs.append(
            probabilities(
                model,
                X
            )
        )


    avg_probs = np.mean(
        all_probs,
        axis=0
    )


    predictions = avg_probs.argmax(
        axis=1
    )


    return predictions, avg_probs


# ============================================================
# TRAIN SISA
# ============================================================

models = []

training_time = 0


for shard_id, indices in enumerate(
    shard_indices
):

    print(
        f"\nTraining Shard {shard_id}"
    )

    X_shard = X_train[
        indices
    ]

    y_shard = y_train[
        indices
    ]


    model = create_model()

    start = time.time()

    model.fit(
        X_shard,
        y_shard
    )

    elapsed = (
        time.time() - start
    )

    training_time += elapsed

    models.append(
        model
    )

    print(
        f"Samples: {len(indices)}"
    )

    print(
        f"Time: {elapsed:.2f} sec"
    )


# ============================================================
# INITIAL EVALUATION
# ============================================================

predictions, probs = aggregate(
    models,
    X_test,
    y_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    "\n" + "=" * 70
)

print("INITIAL SISA RESULTS")

print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# UNLEARNING
# ============================================================

print(
    "\n" + "=" * 70
)

print("SISA UNLEARNING")

print("=" * 70)


affected_shard = FORGET_SHARD

indices = list(
    shard_indices[
        affected_shard
    ]
)


forget_index = indices[
    FORGET_POSITION
]


updated_indices = [
    index
    for position, index
    in enumerate(indices)
    if position != FORGET_POSITION
]


X_updated = X_train[
    updated_indices
]

y_updated = y_train[
    updated_indices
]


print(
    "Affected shard:",
    affected_shard
)

print(
    "Forgotten sample:",
    forget_index
)


# Retrain affected shard from scratch

start = time.time()

new_model = create_model()

new_model.fit(
    X_updated,
    y_updated
)

unlearning_time = (
    time.time() - start
)


models[
    affected_shard
] = new_model


# ============================================================
# POST UNLEARNING
# ============================================================

post_predictions, _ = aggregate(
    models,
    X_test,
    y_test
)


post_accuracy = accuracy_score(
    y_test,
    post_predictions
)


print(
    "\nAfter unlearning:"
)

print(
    f"Accuracy: "
    f"{post_accuracy * 100:.2f}%"
)

print(
    f"Unlearning time: "
    f"{unlearning_time:.2f} sec"
)


print(
    "\nExperiment complete."
)
# The code below is an accidental duplicate of the completed experiment.
raise SystemExit
# ============================================================
# PROBABILITY
# ============================================================

def probabilities(
    model,
    X
):

    return model.predict_proba(X)


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(
    models,
    X,
    y
):

    all_probs = []

    for model in models:

        all_probs.append(
            probabilities(
                model,
                X
            )
        )


    avg_probs = np.mean(
        all_probs,
        axis=0
    )


    predictions = avg_probs.argmax(
        axis=1
    )


    return predictions, avg_probs


# ============================================================
# TRAIN SISA
# ============================================================

models = []

training_time = 0


for shard_id, indices in enumerate(
    shard_indices
):

    print(
        f"\nTraining Shard {shard_id}"
    )

    X_shard = X_train[
        indices
    ]

    y_shard = y_train[
        indices
    ]


    model = create_model()

    start = time.time()

    model.fit(
        X_shard,
        y_shard
    )

    elapsed = (
        time.time() - start
    )

    training_time += elapsed

    models.append(
        model
    )

    print(
        f"Samples: {len(indices)}"
    )

    print(
        f"Time: {elapsed:.2f} sec"
    )


# ============================================================
# INITIAL EVALUATION
# ============================================================

predictions, probs = aggregate(
    models,
    X_test,
    y_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    "\n" + "=" * 70
)

print("INITIAL SISA RESULTS")

print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# UNLEARNING
# ============================================================

print(
    "\n" + "=" * 70
)

print("SISA UNLEARNING")

print("=" * 70)


affected_shard = FORGET_SHARD

indices = list(
    shard_indices[
        affected_shard
    ]
)


forget_index = indices[
    FORGET_POSITION
]


updated_indices = [
    index
    for position, index
    in enumerate(indices)
    if position != FORGET_POSITION
]


X_updated = X_train[
    updated_indices
]

y_updated = y_train[
    updated_indices
]


print(
    "Affected shard:",
    affected_shard
)

print(
    "Forgotten sample:",
    forget_index
)


# Retrain affected shard from scratch

start = time.time()

new_model = create_model()

new_model.fit(
    X_updated,
    y_updated
)

unlearning_time = (
    time.time() - start
)


models[
    affected_shard
] = new_model


# ============================================================
# POST UNLEARNING
# ============================================================

post_predictions, _ = aggregate(
    models,
    X_test,
    y_test
)


post_accuracy = accuracy_score(
    y_test,
    post_predictions
)


print(
    "\nAfter unlearning:"
)

print(
    f"Accuracy: "
    f"{post_accuracy * 100:.2f}%"
)

print(
    f"Unlearning time: "
    f"{unlearning_time:.2f} sec"
)


print(
    "\nExperiment complete."
)
