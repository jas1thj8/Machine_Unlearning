#here in SISA the FINE TUNING will be as the model training on the shards induvidually
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
from sklearn.preprocessing import StandardScaler
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

# Number of SISA shards
NUM_SHARDS = 3

# Separate portion used to create the common base model.
# IMPORTANT:
# The base model never sees the samples used by SISA.
PRETRAIN_RATIO = 0.20

# Training
BASE_EPOCHS = 60
SISA_EPOCHS = 80
FT_EPOCHS = 80

# Unlearning
UNLEARNING_EPOCHS = 10

# Batch size
BATCH_SIZE = 16

# Learning rates
BASE_LR = 0.001
SISA_LR = 0.001
FT_LR = 0.001
UNLEARN_LR = 0.0005

# Regularization
WEIGHT_DECAY = 1e-4

# Forget sample
FORGET_SHARD = 0
FORGET_POSITION = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SISA + FINE-TUNING | BREAST CANCER")
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

print("\nLoading Breast Cancer dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# REMOVE ID / UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = []

for column in df.columns:

    column_lower = column.lower().strip()

    if column_lower in [
        "id",
        "unnamed: 0",
        "unnamed: 32"
    ]:

        columns_to_drop.append(column)


if columns_to_drop:

    df = df.drop(
        columns=columns_to_drop
    )

    print(
        "\nRemoved columns:",
        columns_to_drop
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "diagnosis",
    "target",
    "label",
    "class"
]

target_column = None

for column in df.columns:

    if column.lower().strip() in target_candidates:

        target_column = column
        break


if target_column is None:

    raise ValueError(
        "Target column not found. "
        "Expected diagnosis/target/label/class."
    )


print(
    "\nTarget column:",
    target_column
)


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

y_raw = df[target_column].copy()

X_df = df.drop(
    columns=[target_column]
).copy()


# ============================================================
# CLEAN TARGET
# ============================================================

print("\nOriginal target distribution:")
print(
    y_raw.value_counts(
        dropna=False
    )
)


y_clean = (
    y_raw
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# TARGET MAPPING
# ============================================================

target_mapping = {
    "B": 0,
    "M": 1,
    "BENIGN": 0,
    "MALIGNANT": 1
}


y_series = y_clean.map(
    target_mapping
)


# Check for unknown values

if y_series.isna().any():

    unknown_values = sorted(
        y_clean[
            y_series.isna()
        ].unique()
    )

    raise ValueError(
        f"Unknown target values: {unknown_values}"
    )


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for column in X_df.columns:

    X_df[column] = pd.to_numeric(
        X_df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid_rows = (
    X_df.notna().all(axis=1)
    &
    y_series.notna()
)


X_df = X_df.loc[
    valid_rows
].reset_index(drop=True)


y_series = y_series.loc[
    valid_rows
].reset_index(drop=True)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = X_df.to_numpy(
    dtype=np.float32
)

y = y_series.to_numpy(
    dtype=np.int64
)


print("\n" + "-" * 70)
print("DATASET SUMMARY")
print("-" * 70)

print(
    "Samples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)

unique, counts = np.unique(
    y,
    return_counts=True
)

for cls, count in zip(
    unique,
    counts
):

    label = (
        "Benign"
        if cls == 0
        else "Malignant"
    )

    print(
        f"Class {cls} ({label}): {count}"
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


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit only on training data.
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
# PYTORCH DATASET
# ============================================================

class BreastCancerDataset(
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

        return len(self.X)

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

class BreastCancerMLP(
    nn.Module
):

    def __init__(
        self,
        input_features
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_features,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                32,
                2
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


def create_model():

    return BreastCancerMLP(
        X_train.shape[1]
    ).to(DEVICE)


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

    dataset = BreastCancerDataset(
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
        patience=10
    )

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

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

            running_loss += (
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
            running_loss
            /
            max(1, len(loader))
        )

        epoch_accuracy = (
            100.0
            *
            correct
            /
            total
        )

        scheduler.step(
            epoch_loss
        )

        # Print useful checkpoints
        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
            or
            epoch == epochs - 1
        ):

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"{description} | "
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Train Acc: {epoch_accuracy:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

    elapsed = (
        time.time()
        -
        start_time
    )

    return elapsed


# ============================================================
# EVALUATE SINGLE MODEL
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    name
):

    dataset = BreastCancerDataset(
        X_data,
        y_data
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    predictions_all = []
    probabilities_all = []
    labels_all = []

    with torch.inference_mode():

        for features, labels in loader:

            features = features.to(
                DEVICE
            )

            outputs = model(
                features
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            probabilities_all.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    precision = precision_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    recall = recall_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    f1 = f1_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    auc = roc_auc_score(
        labels_all,
        probabilities_all
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

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
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            labels_all,
            predictions_all
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# STRATIFIED SISA SHARD CREATION
# ============================================================

def create_stratified_shards(
    y_data,
    num_shards
):

    rng = np.random.default_rng(
        SEED
    )

    shard_indices = [
        []
        for _ in range(num_shards)
    ]

    for class_value in np.unique(
        y_data
    ):

        class_indices = np.where(
            y_data == class_value
        )[0]

        rng.shuffle(
            class_indices
        )

        splits = np.array_split(
            class_indices,
            num_shards
        )

        for shard_id in range(
            num_shards
        ):

            shard_indices[
                shard_id
            ].extend(
                splits[shard_id].tolist()
            )

    # Shuffle each final shard
    for shard in shard_indices:

        rng.shuffle(
            shard
        )

    return shard_indices


# ============================================================
# AGGREGATE SISA MODELS
# ============================================================

def aggregate_models(
    models,
    X_data,
    y_data,
    name
):

    dataset = BreastCancerDataset(
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

    predictions_all = []
    probabilities_all = []
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

            # Average probability from all SISA models
            average_probability = torch.stack(
                model_probabilities,
                dim=0
            ).mean(
                dim=0
            )

            predictions = average_probability.argmax(
                dim=1
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            probabilities_all.extend(
                average_probability[:, 1]
                .cpu()
                .numpy()
            )

            labels_all.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )

    precision = precision_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    recall = recall_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    f1 = f1_score(
        labels_all,
        predictions_all,
        zero_division=0
    )

    auc = roc_auc_score(
        labels_all,
        probabilities_all
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

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
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            labels_all,
            predictions_all
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# STEP 1
# CREATE BASE PRETRAINING + SISA DATA
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 1: CREATE BASE/SISA DATA"
)

print(
    "=" * 70
)


all_indices = np.arange(
    len(X_train)
)

rng = np.random.default_rng(
    SEED
)

# Stratified split:
# Base model gets 20%
# SISA gets 80%

pretrain_indices, sisa_indices = train_test_split(
    all_indices,
    test_size=(1.0 - PRETRAIN_RATIO),
    random_state=SEED,
    stratify=y_train
)


X_pretrain = X_train[
    pretrain_indices
]

y_pretrain = y_train[
    pretrain_indices
]

X_sisa = X_train[
    sisa_indices
]

y_sisa = y_train[
    sisa_indices
]


print(
    "Base pretraining samples:",
    len(X_pretrain)
)

print(
    "SISA samples:",
    len(X_sisa)
)


# ============================================================
# STEP 2
# CREATE STRATIFIED SISA SHARDS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 2: CREATE SISA SHARDS"
)

print(
    "=" * 70
)


sisa_shard_indices = create_stratified_shards(
    y_sisa,
    NUM_SHARDS
)


sisa_shards = []


for shard_id, indices in enumerate(
    sisa_shard_indices
):

    X_shard = X_sisa[
        indices
    ]

    y_shard = y_sisa[
        indices
    ]

    sisa_shards.append(
        (
            X_shard,
            y_shard
        )
    )

    print(
        f"\nShard {shard_id}: "
        f"{len(indices)} samples"
    )

    unique, counts = np.unique(
        y_shard,
        return_counts=True
    )

    for cls, count in zip(
        unique,
        counts
    ):

        label = (
            "Benign"
            if cls == 0
            else "Malignant"
        )

        print(
            f"  {label}: {count}"
        )


# ============================================================
# STEP 3
# NORMAL FULL MODEL BASELINE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 3: NORMAL FULL MODEL BASELINE"
)

print(
    "=" * 70
)


normal_model = create_model()


normal_training_time = train_model(
    normal_model,
    X_train,
    y_train,
    BASE_EPOCHS,
    BASE_LR,
    "NORMAL MODEL"
)


normal_results = evaluate_model(
    normal_model,
    X_test,
    y_test,
    "NORMAL FULL MODEL"
)


# ============================================================
# STEP 4
# BASE MODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 4: TRAIN BASE MODEL"
)

print(
    "=" * 70
)


base_model = create_model()


base_training_time = train_model(
    base_model,
    X_pretrain,
    y_pretrain,
    BASE_EPOCHS,
    BASE_LR,
    "BASE MODEL"
)


# ============================================================
# STEP 5
# ORIGINAL SISA
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 5: ORIGINAL SISA"
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
        f"\nTraining SISA Shard {shard_id}"
    )

    model = create_model()

    elapsed = train_model(
        model,
        X_shard,
        y_shard,
        SISA_EPOCHS,
        SISA_LR,
        f"SISA SHARD {shard_id}"
    )

    sisa_training_time += elapsed

    sisa_models.append(
        model
    )


# ============================================================
# STEP 6
# SISA + FINE-TUNING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 6: SISA + FINE-TUNING"
)

print(
    "=" * 70
)


hybrid_models = []

hybrid_training_time = 0.0


for shard_id, (
    X_shard,
    y_shard
) in enumerate(
    sisa_shards
):

    print(
        f"\nFine-tuning Shard {shard_id}"
    )

    # Create model
    model = create_model()

    # Start from common base model
    model.load_state_dict(
        copy.deepcopy(
            base_model.state_dict()
        )
    )

    elapsed = train_model(
        model,
        X_shard,
        y_shard,
        FT_EPOCHS,
        FT_LR,
        f"SISA-FT SHARD {shard_id}"
    )

    hybrid_training_time += elapsed

    hybrid_models.append(
        model
    )


# ============================================================
# STEP 7
# INITIAL COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 7: INITIAL MODEL COMPARISON"
)

print(
    "=" * 70
)


sisa_results = aggregate_models(
    sisa_models,
    X_test,
    y_test,
    "ORIGINAL SISA"
)


hybrid_results = aggregate_models(
    hybrid_models,
    X_test,
    y_test,
    "SISA + FINE-TUNING"
)


# ============================================================
# STEP 8
# SELECT FORGET SAMPLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 8: UNLEARNING REQUEST"
)

print(
    "=" * 70
)


if FORGET_SHARD >= NUM_SHARDS:

    raise ValueError(
        "FORGET_SHARD is outside NUM_SHARDS."
    )


X_forget_shard = sisa_shards[
    FORGET_SHARD
][0]

y_forget_shard = sisa_shards[
    FORGET_SHARD
][1]


if FORGET_POSITION >= len(
    X_forget_shard
):

    raise ValueError(
        "FORGET_POSITION is outside the selected shard."
    )


forget_sample = X_forget_shard[
    FORGET_POSITION
]

forget_label = y_forget_shard[
    FORGET_POSITION
]


print(
    "Affected shard:",
    FORGET_SHARD
)

print(
    "Forget position:",
    FORGET_POSITION
)

print(
    "Forget label:",
    (
        "Malignant"
        if forget_label == 1
        else "Benign"
    )
)


# ============================================================
# FORGET SAMPLE PREDICTION
# ============================================================

def forget_sample_prediction(
    models,
    sample
):

    sample_tensor = torch.tensor(
        sample,
        dtype=torch.float32
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    probabilities = []

    for model in models:

        model.eval()

        with torch.inference_mode():

            outputs = model(
                sample_tensor
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

        probabilities.append(
            probs
        )

    average_probability = torch.stack(
        probabilities
    ).mean(
        dim=0
    )

    prediction = average_probability.argmax(
        dim=1
    ).item()

    confidence = average_probability[
        0,
        prediction
    ].item()

    return prediction, confidence


before_prediction, before_confidence = (
    forget_sample_prediction(
        hybrid_models,
        forget_sample
    )
)


print(
    "\nForget sample BEFORE unlearning:"
)

print(
    "Predicted:",
    (
        "Malignant"
        if before_prediction == 1
        else "Benign"
    )
)

print(
    f"Confidence: "
    f"{before_confidence * 100:.2f}%"
)


# ============================================================
# STEP 9
# REMOVE FORGET SAMPLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 9: REMOVE FORGET SAMPLE"
)

print(
    "=" * 70
)


keep_mask = np.ones(
    len(X_forget_shard),
    dtype=bool
)

keep_mask[
    FORGET_POSITION
] = False


X_updated_shard = X_forget_shard[
    keep_mask
]

y_updated_shard = y_forget_shard[
    keep_mask
]


print(
    "Original shard size:",
    len(X_forget_shard)
)

print(
    "Updated shard size:",
    len(X_updated_shard)
)


# ============================================================
# STEP 10
# SISA-FT UNLEARNING
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 10: SISA-FT UNLEARNING"
)

print(
    "=" * 70
)


affected_model = hybrid_models[
    FORGET_SHARD
]


unlearning_start = time.time()


train_model(
    affected_model,
    X_updated_shard,
    y_updated_shard,
    UNLEARNING_EPOCHS,
    UNLEARN_LR,
    "SISA-FT UNLEARNING"
)


unlearning_time = (
    time.time()
    -
    unlearning_start
)


hybrid_models[
    FORGET_SHARD
] = affected_model


# ============================================================
# STEP 11
# POST-UNLEARNING EVALUATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 11: POST-UNLEARNING EVALUATION"
)

print(
    "=" * 70
)


post_results = aggregate_models(
    hybrid_models,
    X_test,
    y_test,
    "SISA-FT AFTER UNLEARNING"
)


# ============================================================
# FORGET SAMPLE AFTER UNLEARNING
# ============================================================

after_prediction, after_confidence = (
    forget_sample_prediction(
        hybrid_models,
        forget_sample
    )
)


print(
    "\nForget sample AFTER unlearning:"
)

print(
    "Predicted:",
    (
        "Malignant"
        if after_prediction == 1
        else "Benign"
    )
)

print(
    f"Confidence: "
    f"{after_confidence * 100:.2f}%"
)


confidence_change = (
    after_confidence
    -
    before_confidence
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL EXPERIMENT SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nAccuracy"
)

print(
    f"Normal Model       : "
    f"{normal_results['accuracy'] * 100:.2f}%"
)

print(
    f"Original SISA      : "
    f"{sisa_results['accuracy'] * 100:.2f}%"
)

print(
    f"SISA + Fine-Tuning : "
    f"{hybrid_results['accuracy'] * 100:.2f}%"
)

print(
    f"After Unlearning   : "
    f"{post_results['accuracy'] * 100:.2f}%"
)


print(
    "\nHybrid improvement over SISA:"
)

print(
    f"{(hybrid_results['accuracy'] - sisa_results['accuracy']) * 100:+.2f} "
    f"percentage points"
)


print(
    "\nAccuracy change after unlearning:"
)

print(
    f"{(post_results['accuracy'] - hybrid_results['accuracy']) * 100:+.2f} "
    f"percentage points"
)


print(
    "\nForget-sample confidence:"
)

print(
    f"Before: "
    f"{before_confidence * 100:.2f}%"
)

print(
    f"After : "
    f"{after_confidence * 100:.2f}%"
)

print(
    f"Change: "
    f"{confidence_change * 100:+.2f} percentage points"
)


print(
    "\nTraining time:"
)

print(
    f"Normal Model      : "
    f"{normal_training_time:.2f} sec"
)

print(
    f"Base Model        : "
    f"{base_training_time:.2f} sec"
)

print(
    f"Original SISA     : "
    f"{sisa_training_time:.2f} sec"
)

print(
    f"SISA + Fine-Tune  : "
    f"{hybrid_training_time:.2f} sec"
)

print(
    f"Unlearning        : "
    f"{unlearning_time:.2f} sec"
)


print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)