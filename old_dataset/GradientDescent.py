import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET = "Y:\\VSc\\MUL\\breast-cancer.csv"

RANDOM_STATE = 42
FORGET_RATIO = 0.10

EPOCHS = 100
LEARNING_RATE = 0.001

# Unlearning strength
UNLEARNING_RATE = 0.001

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)


# ============================================================
# 3. CLEAN DATASET
# ============================================================

# Remove unnecessary columns
columns_to_remove = []

for column in ["id", "ID", "Unnamed: 32"]:
    if column in df.columns:
        columns_to_remove.append(column)

df = df.drop(columns=columns_to_remove)


# Convert target
if "diagnosis" in df.columns:

    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

else:
    raise ValueError(
        "Column 'diagnosis' was not found."
    )


# Features and target
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 5. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to tensors
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)


print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. CREATE MLP MODEL
# ============================================================

class CancerMLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


input_size = X_train.shape[1]


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_model(model, X, y, epochs=100):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    start_time = time.time()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

    training_time = time.time() - start_time

    return training_time


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):

    model.eval()

    with torch.no_grad():

        outputs = model(X)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).float()

    accuracy = accuracy_score(
        y.numpy(),
        predictions.numpy()
    )

    f1 = f1_score(
        y.numpy(),
        predictions.numpy()
    )

    return accuracy, f1


# ============================================================
# 9. TRAIN ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ORIGINAL MODEL")
print("=" * 60)

original_model = CancerMLP(
    input_size
)

original_training_time = train_model(
    original_model,
    X_train,
    y_train,
    EPOCHS
)

original_accuracy, original_f1 = evaluate_model(
    original_model,
    X_test,
    y_test
)

print(
    f"Training time: "
    f"{original_training_time:.4f} seconds"
)

print(
    f"Accuracy: "
    f"{original_accuracy:.4f}"
)

print(
    f"F1 Score: "
    f"{original_f1:.4f}"
)


# ============================================================
# 10. CREATE FORGET SET
# ============================================================

number_of_forget_samples = int(
    len(X_train) * FORGET_RATIO
)

rng = np.random.default_rng(
    RANDOM_STATE
)

forget_indices = rng.choice(
    len(X_train),
    size=number_of_forget_samples,
    replace=False
)

forget_indices = np.sort(
    forget_indices
)

print("\n" + "=" * 60)
print("FORGET SET")
print("=" * 60)

print(
    "Forget samples:",
    len(forget_indices)
)


# ============================================================
# 11. CREATE FORGET DATA
# ============================================================

X_forget = X_train[
    forget_indices
]

y_forget = y_train[
    forget_indices
]


# ============================================================
# 12. COPY ORIGINAL MODEL
# ============================================================

unlearned_model = copy.deepcopy(
    original_model
)


# ============================================================
# 13. GRADIENT-BASED UNLEARNING
# ============================================================

print("\n" + "=" * 60)
print("GRADIENT-BASED UNLEARNING")
print("=" * 60)

criterion = nn.BCEWithLogitsLoss()

start_time = time.time()


unlearned_model.train()


# Clear existing gradients
unlearned_model.zero_grad()


# ------------------------------------------------------------
# Calculate gradient generated by forget samples
# ------------------------------------------------------------

forget_outputs = unlearned_model(
    X_forget
)

forget_loss = criterion(
    forget_outputs,
    y_forget
)

forget_loss.backward()


# ------------------------------------------------------------
# Modify parameters
# ------------------------------------------------------------

with torch.no_grad():

    for parameter in unlearned_model.parameters():

        if parameter.grad is not None:

            # Move parameters in the direction
            # intended to reduce the contribution
            # of the forget samples.

            parameter += (
                UNLEARNING_RATE *
                parameter.grad
            )


unlearning_time = (
    time.time() - start_time
)


# ============================================================
# 14. EVALUATE UNLEARNED MODEL
# ============================================================

unlearned_accuracy, unlearned_f1 = evaluate_model(
    unlearned_model,
    X_test,
    y_test
)

print(
    f"Unlearning time: "
    f"{unlearning_time:.4f} seconds"
)

print(
    f"Accuracy: "
    f"{unlearned_accuracy:.4f}"
)

print(
    f"F1 Score: "
    f"{unlearned_f1:.4f}"
)


# ============================================================
# 15. FULL RETRAINING BASELINE
# ============================================================

print("\n" + "=" * 60)
print("FULL RETRAINING")
print("=" * 60)


retain_mask = np.ones(
    len(X_train),
    dtype=bool
)

retain_mask[
    forget_indices
] = False


X_retain = X_train[
    retain_mask
]

y_retain = y_train[
    retain_mask
]


retrained_model = CancerMLP(
    input_size
)


retraining_time = train_model(
    retrained_model,
    X_retain,
    y_retain,
    EPOCHS
)


retrained_accuracy, retrained_f1 = evaluate_model(
    retrained_model,
    X_test,
    y_test
)


print(
    f"Retraining time: "
    f"{retraining_time:.4f} seconds"
)

print(
    f"Accuracy: "
    f"{retrained_accuracy:.4f}"
)

print(
    f"F1 Score: "
    f"{retrained_f1:.4f}"
)


# ============================================================
# 16. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

results = pd.DataFrame({

    "Model": [
        "Original Model",
        "Gradient Unlearned",
        "Full Retraining"
    ],

    "Accuracy": [
        original_accuracy,
        unlearned_accuracy,
        retrained_accuracy
    ],

    "F1 Score": [
        original_f1,
        unlearned_f1,
        retrained_f1
    ],

    "Time (seconds)": [
        original_training_time,
        unlearning_time,
        retraining_time
    ]
})


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 17. SPEEDUP
# ============================================================

if unlearning_time > 0:

    speedup = (
        retraining_time /
        unlearning_time
    )

    print(
        f"\nUnlearning speedup: "
        f"{speedup:.2f}x"
    )


# ============================================================
# 18. MODEL PARAMETER DIFFERENCE
# ============================================================

total_parameter_change = 0.0

for original_parameter, unlearned_parameter in zip(
    original_model.parameters(),
    unlearned_model.parameters()
):

    difference = torch.norm(
        original_parameter -
        unlearned_parameter
    )

    total_parameter_change += (
        difference.item()
    )


print(
    "\nTotal parameter change:",
    total_parameter_change
)
