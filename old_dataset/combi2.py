import os
import time
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

CSV_PATH = r"Y:\\VSc\\MUL\\breast-cancer.csv"

# CHANGE THIS if your target column has a known name.
# Example:
# TARGET_COLUMN = "failure"
# TARGET_COLUMN = "battery_failure"
TARGET_COLUMN = None

NUM_SHARDS = 3

PRETRAIN_EPOCHS = 60
SISA_EPOCHS = 80

BATCH_SIZE = 16

PRETRAIN_LR = 0.001
SISA_LR = 0.001

WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRETRAINED MODEL + SISA")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print("\nColumns:")

for column in df.columns:

    print(
        " ",
        column
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

if TARGET_COLUMN is None:

    target_candidates = [
        "target",
        "label",
        "class",
        "failure",
        "battery_failure",
        "failure_label",
        "status",
        "diagnosis"
    ]

    TARGET_COLUMN = None

    for column in df.columns:

        if column.lower().strip() in target_candidates:

            TARGET_COLUMN = column

            break


if TARGET_COLUMN is None:

    raise ValueError(
        "\nCould not automatically identify "
        "the target column.\n\n"
        "Set TARGET_COLUMN manually at the "
        "top of the program."
    )


print(
    "\nTarget column:",
    TARGET_COLUMN
)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = []

for column in df.columns:

    name = column.lower().strip()

    if name in [
        "id",
        "unnamed: 0",
        "unnamed: 32"
    ]:

        columns_to_drop.append(
            column
        )


if columns_to_drop:

    df = df.drop(
        columns=columns_to_drop
    )

    print(
        "\nRemoved:",
        columns_to_drop
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

y_raw = df[
    TARGET_COLUMN
].copy()

X_df = df.drop(
    columns=[
        TARGET_COLUMN
    ]
).copy()


# ============================================================
# CONVERT CATEGORICAL FEATURES
# ============================================================

print(
    "\nPreparing features..."
)

for column in X_df.columns:

    if not pd.api.types.is_numeric_dtype(
        X_df[column]
    ):

        X_df[column] = (
            X_df[column]
            .astype(str)
            .str.strip()
        )

        encoder = LabelEncoder()

        X_df[column] = encoder.fit_transform(
            X_df[column]
        )


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid_rows = (
    X_df.notna().all(axis=1)
    &
    y_raw.notna()
)


X_df = X_df.loc[
    valid_rows
].reset_index(
    drop=True
)

y_raw = y_raw.loc[
    valid_rows
].reset_index(
    drop=True
)


# ============================================================
# ENCODE TARGET
# ============================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y_raw.astype(str)
)


X = X_df.to_numpy(
    dtype=np.float32
)

y = y.astype(
    np.int64
)


NUM_CLASSES = len(
    np.unique(y)
)

NUM_FEATURES = X.shape[1]


print("\n" + "-" * 70)
print("DATASET INFORMATION")
print("-" * 70)

print(
    "Samples:",
    len(X)
)

print(
    "Features:",
    NUM_FEATURES
)

print(
    "Classes:",
    NUM_CLASSES
)

print(
    "\nTarget classes:"
)

for index, label in enumerate(
    target_encoder.classes_
):

    count = np.sum(
        y == index
    )

    print(
        f"  {index}: {label} "
        f"({count} samples)"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
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


X_train = X_train.astype(
    np.float32
)

X_test = X_test.astype(
    np.float32
)


# ============================================================
# DATASET CLASS
# ============================================================

class BatteryDataset(
    Dataset
):

    def __init__(
        self,
        X_data,
        y_data
    ):

        self.X = torch.tensor(
            X_data,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y_data,
            dtype=torch.long
        )

    def __len__(self):

        return len(
            self.X
        )

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# MODEL
# ============================================================

class BatteryMLP(
    nn.Module
):

    def __init__(
        self,
        input_features,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                128
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                128
            ),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                64
            ),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(x)


def create_model():

    return BatteryMLP(
        NUM_FEATURES,
        NUM_CLASSES
    ).to(
        DEVICE
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X_data,
    y_data,
    epochs,
    learning_rate,
    description
):

    dataset = BatteryDataset(
        X_data,
        y_data
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    start = time.time()

    for epoch in range(
        epochs
    ):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for features, labels in loader:

            features = features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        epoch_loss = (
            total_loss
            /
            max(1, len(loader))
        )

        epoch_accuracy = (
            100
            *
            correct
            /
            total
        )

        scheduler.step(
            epoch_loss
        )


        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
            or
            epoch == epochs - 1
        ):

            lr = optimizer.param_groups[
                0
            ]["lr"]

            print(
                f"{description} | "
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_accuracy:.2f}% | "
                f"LR: {lr:.6f}"
            )


    return time.time() - start


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    name
):

    dataset = BatteryDataset(
        X_data,
        y_data
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    predictions = []

    probabilities = []

    labels_all = []


    with torch.inference_mode():

        for features, labels in loader:

            features = features.to(
                DEVICE
            )

            outputs = model(
                features
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            pred = probs.argmax(
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )


    predictions = np.array(
        predictions
    )

    probabilities = np.array(
        probabilities
    )

    labels_all = np.array(
        labels_all
    )


    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    precision = precision_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        "\n" + "-" * 70
    )

    print(
        name
    )

    print(
        "-" * 70
    )

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            labels_all,
            predictions
        )
    )


    return accuracy


# ============================================================
# CREATE SISA SHARDS
# ============================================================

def create_sisa_shards(
    X_data,
    y_data,
    num_shards
):

    rng = np.random.default_rng(
        SEED
    )

    shard_indices = [
        []
        for _ in range(
            num_shards
        )
    ]


    # Stratify each class
    for class_value in np.unique(
        y_data
    ):

        indices = np.where(
            y_data == class_value
        )[0]

        rng.shuffle(
            indices
        )

        splits = np.array_split(
            indices,
            num_shards
        )


        for shard_id in range(
            num_shards
        ):

            shard_indices[
                shard_id
            ].extend(
                splits[
                    shard_id
                ].tolist()
            )


    # Shuffle final shards
    for shard in shard_indices:

        rng.shuffle(
            shard
        )


    return shard_indices


# ============================================================
# STEP 1
# TRAIN PRETRAINED MODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 1: TRAIN PRETRAINED MODEL"
)

print(
    "=" * 70
)


pretrained_model = create_model()


pretraining_time = train_model(
    pretrained_model,
    X_train,
    y_train,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    "PRETRAINED MODEL"
)


evaluate_model(
    pretrained_model,
    X_test,
    y_test,
    "PRETRAINED MODEL"
)


# ============================================================
# STEP 2
# CREATE SISA SHARDS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 2: SISA SHARDING"
)

print(
    "=" * 70
)


shard_indices = create_sisa_shards(
    X_train,
    y_train,
    NUM_SHARDS
)


sisa_shards = []


for shard_id, indices in enumerate(
    shard_indices
):

    X_shard = X_train[
        indices
    ]

    y_shard = y_train[
        indices
    ]

    sisa_shards.append(
        (
            X_shard,
            y_shard
        )
    )


    print(
        f"\nShard {shard_id}"
    )

    print(
        "Samples:",
        len(X_shard)
    )

    unique, counts = np.unique(
        y_shard,
        return_counts=True
    )

    for cls, count in zip(
        unique,
        counts
    ):

        print(
            f"  Class {cls}: {count}"
        )


# ============================================================
# STEP 3
# CREATE CHILD MODELS FROM PRETRAINED MODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 3: SISA CHILD MODELS"
)

print(
    "=" * 70
)


sisa_models = []

sisa_training_time = 0.0


for shard_id, (
    X_shard,
    y_shard
) in enumerate(
    sisa_shards
):


    print(
        f"\nCreating child model "
        f"for Shard {shard_id}"
    )


    # --------------------------------------------------------
    # CREATE CHILD MODEL
    # --------------------------------------------------------

    child_model = create_model()


    # --------------------------------------------------------
    # COPY PRETRAINED MODEL WEIGHTS
    # --------------------------------------------------------

    child_model.load_state_dict(
        copy.deepcopy(
            pretrained_model.state_dict()
        )
    )


    print(
        "Pretrained weights copied."
    )


    # --------------------------------------------------------
    # TRAIN CHILD MODEL ONLY ON ITS SHARD
    # --------------------------------------------------------

    elapsed = train_model(
        child_model,
        X_shard,
        y_shard,
        SISA_EPOCHS,
        SISA_LR,
        f"SISA SHARD {shard_id}"
    )


    sisa_training_time += elapsed


    sisa_models.append(
        child_model
    )


# ============================================================
# STEP 4
# AGGREGATE SISA MODELS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 4: SISA AGGREGATION"
)

print(
    "=" * 70
)


def aggregate_predictions(
    models,
    X_data,
    y_data
):

    dataset = BatteryDataset(
        X_data,
        y_data
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    for model in models:

        model.eval()


    predictions = []

    labels_all = []


    with torch.inference_mode():

        for features, labels in loader:

            features = features.to(
                DEVICE
            )


            model_probabilities = []


            for model in models:

                outputs = model(
                    features
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                model_probabilities.append(
                    probabilities
                )


            # Average predictions
            average_probability = torch.stack(
                model_probabilities
            ).mean(
                dim=0
            )


            pred = average_probability.argmax(
                dim=1
            )


            predictions.extend(
                pred.cpu().numpy()
            )

            labels_all.extend(
                labels.numpy()
            )


    predictions = np.array(
        predictions
    )

    labels_all = np.array(
        labels_all
    )


    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    precision = precision_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        "\n" + "-" * 70
    )

    print(
        "PRETRAINED + SISA"
    )

    print(
        "-" * 70
    )

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            labels_all,
            predictions
        )
    )


    return accuracy


sisa_accuracy = aggregate_predictions(
    sisa_models,
    X_test,
    y_test
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 70
)


print(
    f"\nPretrained model "
    f"training time: "
    f"{pretraining_time:.2f} sec"
)

print(
    f"SISA child-model "
    f"training time: "
    f"{sisa_training_time:.2f} sec"
)

print(
    f"\nFinal "
    f"Pretrained + SISA accuracy: "
    f"{sisa_accuracy * 100:.2f}%"
)


print(
    "\nArchitecture:"
)

print(
    "Pretrained Model"
)

print(
    "       ↓"
)

print(
    "SISA Sharding"
)

print(
    "       ↓"
)

print(
    "Shard 1 → Child Model 1"
)

print(
    "Shard 2 → Child Model 2"
)

print(
    "Shard 3 → Child Model 3"
)

print(
    "       ↓"
)

print(
    "Probability Aggregation"
)

print(
    "       ↓"
)

print(
    "Final SISA Model"
)

print(
    "\nExperiment complete."
)