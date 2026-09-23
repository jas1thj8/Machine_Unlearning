# ============================================================
# SUPER MODEL → CHILD MODEL → FINE-TUNING UNLEARNING
# Wisconsin Breast Cancer Diagnostic Dataset
# ============================================================

import time
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CSV_PATH = r"Y:\VSc\MUL\breast-cancer.csv"

FORGET_RATIO = 0.10

SUPER_EPOCHS = 100
FINE_TUNE_EPOCHS = 30
RETRAIN_EPOCHS = 100

BATCH_SIZE = 16

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SUPER MODEL → CHILD MODEL FINE-TUNING UNLEARNING")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# REMOVE ID COLUMNS
# ============================================================

drop_columns = []

for col in df.columns:

    name = col.strip().lower()

    if name in [
        "id",
        "unnamed: 0",
        "unnamed: 32"
    ]:
        drop_columns.append(col)


if drop_columns:

    print(
        f"Removing ID columns: {drop_columns}"
    )

    df = df.drop(
        columns=drop_columns
    )


# ============================================================
# TARGET
# ============================================================

target_column = None

for col in df.columns:

    if col.strip().lower() in [
        "diagnosis",
        "target",
        "label",
        "class"
    ]:

        target_column = col
        break


if target_column is None:

    raise ValueError(
        "Target column not found."
    )


print(
    f"Target column: {target_column}"
)

# ============================================================
# TARGET ENCODING + FEATURES
# ============================================================

# Clean target values first
df[target_column] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Wisconsin Breast Cancer Dataset:
# B = Benign = 0
# M = Malignant = 1

if set(df[target_column].unique()).issubset({"B", "M"}):

    df[target_column] = df[target_column].map({
        "B": 0,
        "M": 1
    })

else:

    # Generic categorical target handling
    classes = sorted(
        df[target_column].dropna().unique()
    )

    mapping = {
        value: index
        for index, value in enumerate(classes)
    }

    print(
        f"Target mapping: {mapping}"
    )

    df[target_column] = (
        df[target_column]
        .map(mapping)
    )


# ============================================================
# FEATURES
# ============================================================

X_df = df.drop(
    columns=[target_column]
).copy()

# Convert every feature to numeric
for col in X_df.columns:

    X_df[col] = pd.to_numeric(
        X_df[col],
        errors="coerce"
    )


# Target
y = df[target_column].values


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid_mask = (
    X_df.notna().all(axis=1)
    & pd.notna(y)
)

X_df = X_df.loc[valid_mask]

y = y[valid_mask]


# Convert to NumPy
X = X_df.values.astype(
    np.float32
)

y = np.asarray(
    y,
    dtype=np.int64
)


print(
    f"\nFinal dataset shape: {X.shape}"
)

print(
    f"Number of features: {X.shape[1]}"
)

print(
    f"Target classes: {np.unique(y)}"
)

print(
    f"Class distribution: "
    f"{np.bincount(y)}"
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


print("\nTrain/Test Split")
print("-" * 40)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
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
# TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# MODEL
# ============================================================

class BreastCancerMLP(
    nn.Module
):

    def __init__(
        self,
        input_size,
        num_classes=2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
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
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X,
    y,
    epochs,
    lr,
    name
):

    model = model.to(DEVICE)

    dataset = TensorDataset(
        X,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    start = time.perf_counter()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                batch_X
            )

            loss = criterion(
                output,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == epochs - 1
        ):

            avg_loss = (
                total_loss
                / len(loader)
            )

            print(
                f"{name} | "
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {avg_loss:.6f}"
            )


    elapsed = (
        time.perf_counter()
        - start
    )

    return model, elapsed


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X,
    y,
    name
):

    model.eval()

    with torch.no_grad():

        X_device = X.to(
            DEVICE
        )

        output = model(
            X_device
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        ).cpu().numpy()


    y_true = y.numpy()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# PARAMETER DISTANCE
# ============================================================

def parameter_distance(
    model_a,
    model_b
):

    total = 0.0

    for p1, p2 in zip(
        model_a.parameters(),
        model_b.parameters()
    ):

        total += torch.sum(
            (p1.detach() - p2.detach()) ** 2
        ).item()


    return np.sqrt(total)


# ============================================================
# PHASE 1
# TRAIN SUPER MODEL
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 1 — TRAIN SUPER MODEL")
print("=" * 70)

print(
    f"\nTraining Super Model on "
    f"ALL {len(X_train)} training samples."
)


super_model = BreastCancerMLP(
    input_size=X_train.shape[1]
)


super_model, super_time = train_model(
    super_model,
    X_train_tensor,
    y_train_tensor,
    SUPER_EPOCHS,
    LEARNING_RATE,
    "[SUPER]"
)


super_results = evaluate(
    super_model,
    X_test_tensor,
    y_test_tensor,
    "SUPER MODEL"
)


# ============================================================
# PHASE 2
# CREATE FORGET SET
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 2 — CREATE FORGET SET")
print("=" * 70)


num_samples = len(
    X_train
)

num_forget = int(
    num_samples * FORGET_RATIO
)


rng = np.random.default_rng(
    SEED
)


forget_indices = rng.choice(
    num_samples,
    size=num_forget,
    replace=False
)


forget_indices = np.sort(
    forget_indices
)


retain_mask = np.ones(
    num_samples,
    dtype=bool
)

retain_mask[
    forget_indices
] = False


retain_indices = np.where(
    retain_mask
)[0]


X_forget = X_train_tensor[
    forget_indices
]

y_forget = y_train_tensor[
    forget_indices
]


X_retain = X_train_tensor[
    retain_indices
]

y_retain = y_train_tensor[
    retain_indices
]


print(
    f"\nTotal training samples : "
    f"{num_samples}"
)

print(
    f"Forget ratio           : "
    f"{FORGET_RATIO * 100:.1f}%"
)

print(
    f"Forgotten samples      : "
    f"{len(X_forget)}"
)

print(
    f"Retained samples       : "
    f"{len(X_retain)}"
)

print(
    "\nForgotten training indices:"
)

print(
    forget_indices
)


# ============================================================
# PHASE 3
# CREATE CHILD MODEL
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 — CREATE CHILD MODEL")
print("=" * 70)


child_model = BreastCancerMLP(
    input_size=X_train.shape[1]
)


child_model.load_state_dict(
    copy.deepcopy(
        super_model.state_dict()
    )
)


print(
    "\nChild Model initialized "
    "from Super Model."
)


print(
    f"Initial parameter distance: "
    f"{parameter_distance(super_model, child_model):.10f}"
)


# ============================================================
# PHASE 4
# FINE-TUNE CHILD WITHOUT FORGET DATA
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 4 — FINE-TUNE CHILD MODEL")
print("=" * 70)


print(
    f"\nFine-tuning on ONLY "
    f"{len(X_retain)} retained samples."
)

print(
    f"Excluded from fine-tuning: "
    f"{len(X_forget)} forgotten samples."
)


child_model, fine_tune_time = train_model(
    child_model,
    X_retain,
    y_retain,
    FINE_TUNE_EPOCHS,
    LEARNING_RATE,
    "[CHILD]"
)


child_results = evaluate(
    child_model,
    X_test_tensor,
    y_test_tensor,
    "UNLEARNED CHILD MODEL"
)


# ============================================================
# CHILD PARAMETER CHANGE
# ============================================================

child_change = parameter_distance(
    super_model,
    child_model
)


print(
    f"\nChild parameter change "
    f"from Super Model: "
    f"{child_change:.6f}"
)


# ============================================================
# PHASE 5
# FULL RETRAINING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 5 — FULL RETRAINING REFERENCE")
print("=" * 70)


print(
    f"\nTraining NEW model from scratch "
    f"using {len(X_retain)} retained samples."
)


retrained_model = BreastCancerMLP(
    input_size=X_train.shape[1]
)


retrained_model, retrain_time = train_model(
    retrained_model,
    X_retain,
    y_retain,
    RETRAIN_EPOCHS,
    LEARNING_RATE,
    "[RETRAIN]"
)


retrained_results = evaluate(
    retrained_model,
    X_test_tensor,
    y_test_tensor,
    "FULL RETRAINED MODEL"
)


# ============================================================
# PHASE 6
# FORGOTTEN DATA ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 6 — FORGOTTEN SAMPLE ANALYSIS")
print("=" * 70)


super_model.eval()
child_model.eval()


with torch.no_grad():

    forget_X = X_forget.to(
        DEVICE
    )

    super_output = super_model(
        forget_X
    )

    child_output = child_model(
        forget_X
    )


    super_pred = torch.argmax(
        super_output,
        dim=1
    ).cpu().numpy()


    child_pred = torch.argmax(
        child_output,
        dim=1
    ).cpu().numpy()


forget_true = y_forget.numpy()


super_forget_accuracy = (
    accuracy_score(
        forget_true,
        super_pred
    )
)


child_forget_accuracy = (
    accuracy_score(
        forget_true,
        child_pred
    )
)


print(
    f"\nSuper Model accuracy "
    f"on forgotten samples: "
    f"{super_forget_accuracy * 100:.2f}%"
)


print(
    f"Unlearned Child accuracy "
    f"on forgotten samples: "
    f"{child_forget_accuracy * 100:.2f}%"
)


# ============================================================
# PHASE 7
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)


results = pd.DataFrame({

    "Model": [
        "Super Model",
        "Unlearned Child",
        "Full Retrained"
    ],

    "Accuracy": [
        super_results["accuracy"] * 100,
        child_results["accuracy"] * 100,
        retrained_results["accuracy"] * 100
    ],

    "Precision": [
        super_results["precision"] * 100,
        child_results["precision"] * 100,
        retrained_results["precision"] * 100
    ],

    "Recall": [
        super_results["recall"] * 100,
        child_results["recall"] * 100,
        retrained_results["recall"] * 100
    ],

    "F1": [
        super_results["f1"] * 100,
        child_results["f1"] * 100,
        retrained_results["f1"] * 100
    ],

    "Time_seconds": [
        super_time,
        fine_tune_time,
        retrain_time
    ]
})


print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DIFFERENCES
# ============================================================

accuracy_difference = (
    child_results["accuracy"]
    - retrained_results["accuracy"]
) * 100


f1_difference = (
    child_results["f1"]
    - retrained_results["f1"]
) * 100


print("\n" + "-" * 70)

print(
    "UNLEARNED CHILD vs FULL RETRAINING"
)

print(
    f"Accuracy difference: "
    f"{accuracy_difference:+.2f} percentage points"
)

print(
    f"F1 difference: "
    f"{f1_difference:+.2f} percentage points"
)


# ============================================================
# TIME COMPARISON
# ============================================================

print("\n" + "-" * 70)

print("TIME COMPARISON")

print(
    f"Super Model training : "
    f"{super_time:.4f} seconds"
)

print(
    f"Child fine-tuning    : "
    f"{fine_tune_time:.4f} seconds"
)

print(
    f"Full retraining      : "
    f"{retrain_time:.4f} seconds"
)


if fine_tune_time > 0:

    speedup = (
        retrain_time
        / fine_tune_time
    )

    print(
        f"\nFine-Tuning / "
        f"Full-Retraining ratio: "
        f"{speedup:.2f}x"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print(
    f"""
Dataset:
    Wisconsin Breast Cancer Diagnostic

Training samples:
    {len(X_train)}

Testing samples:
    {len(X_test)}

Features:
    {X_train.shape[1]}

Forgotten samples:
    {len(X_forget)}

Retained samples:
    {len(X_retain)}

Feature columns removed:
    0

SUPER MODEL:
    Trained on all {len(X_train)} samples.

CHILD MODEL:
    Initialized from Super Model.

UNLEARNING:
    Forgotten samples excluded from
    Child Model fine-tuning.

UNLEARNED CHILD:
    Fine-tuned using {len(X_retain)}
    retained samples.

FULL RETRAINING:
    New model trained from scratch
    using retained samples.

Super Model Accuracy:
    {super_results["accuracy"] * 100:.2f}%

Unlearned Child Accuracy:
    {child_results["accuracy"] * 100:.2f}%

Full Retrained Accuracy:
    {retrained_results["accuracy"] * 100:.2f}%

Child vs Full Retraining Accuracy:
    {accuracy_difference:+.2f} pp

Super Model Time:
    {super_time:.4f} s

Child Fine-Tuning Time:
    {fine_tune_time:.4f} s

Full Retraining Time:
    {retrain_time:.4f} s
"""
)

print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)