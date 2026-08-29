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

ORIGINAL_EPOCHS = 100

FINETUNE_EPOCHS = 30

RETRAIN_EPOCHS = 100

LEARNING_RATE = 0.001

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


# ============================================================
# 4. TARGET
# ============================================================

if "diagnosis" not in df.columns:

    raise ValueError(
        "Column 'diagnosis' not found."
    )


# B = Benign
# M = Malignant

df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})


X = df.drop(
    columns=["diagnosis"]
).values

y = df["diagnosis"].values


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

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
# 6. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================================

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


# ============================================================
# 8. MLP MODEL
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
# 9. TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X,
    y,
    epochs,
    learning_rate=LEARNING_RATE
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
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

    elapsed_time = (
        time.time() - start_time
    )

    return elapsed_time


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

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
# 11. TRAIN ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 60)

print(
    "1. TRAINING ORIGINAL MODEL"
)

print("=" * 60)


original_model = CancerMLP(
    input_size
)


original_training_time = train_model(

    original_model,

    X_train,

    y_train,

    ORIGINAL_EPOCHS

)


original_accuracy, original_f1 = (
    evaluate_model(
        original_model,
        X_test,
        y_test
    )
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
# 12. CREATE FORGET SET
# ============================================================
number_of_forget_samples = int(
    len(X_train)
    * FORGET_RATIO
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
print(
    "2. CREATING FORGET SET"
)
print("=" * 60)
print(
    "Forget samples:",
    len(forget_indices)
)
# ============================================================
# 13. CREATE RETAIN SET
# ============================================================

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


X_forget = X_train[
    forget_indices
]

y_forget = y_train[
    forget_indices
]


print(
    "Retain samples:",
    len(X_retain)
)

print(
    "Forget samples:",
    len(X_forget)
)


# ============================================================
# 14. FINE-TUNING UNLEARNING
# ============================================================

print("\n" + "=" * 60)

print(
    "3. FINE-TUNING UNLEARNING"
)

print("=" * 60)


# Make a copy of the original model

unlearned_model = copy.deepcopy(
    original_model
)


# Start timing

start_time = time.time()


# Fine-tune using ONLY retain data

finetuning_time = train_model(

    unlearned_model,

    X_retain,

    y_retain,

    FINETUNE_EPOCHS,

    learning_rate=LEARNING_RATE

)


finetuning_time = (
    time.time() - start_time
)


# ============================================================
# 15. EVALUATE UNLEARNED MODEL
# ============================================================

unlearned_accuracy, unlearned_f1 = (
    evaluate_model(

        unlearned_model,

        X_test,

        y_test

    )
)


print(
    f"Fine-tuning time: "
    f"{finetuning_time:.4f} seconds"
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
# 16. FULL RETRAINING BASELINE
# ============================================================

print("\n" + "=" * 60)

print(
    "4. FULL RETRAINING"
)

print("=" * 60)


retrained_model = CancerMLP(
    input_size
)


retraining_time = train_model(

    retrained_model,

    X_retain,

    y_retain,

    RETRAIN_EPOCHS

)


retrained_accuracy, retrained_f1 = (
    evaluate_model(

        retrained_model,

        X_test,

        y_test

    )
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
# 17. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)

print(
    "FINAL COMPARISON"
)

print("=" * 60)


results = pd.DataFrame({

    "Model": [

        "Original Model",

        "Fine-Tuned Unlearned",

        "Full Retrained"

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

        finetuning_time,

        retraining_time

    ]

})


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 18. SPEEDUP
# ============================================================

if finetuning_time > 0:

    speedup = (

        retraining_time
        /
        finetuning_time

    )

    print(
        f"\nFine-tuning speedup: "
        f"{speedup:.2f}x"
    )


# ============================================================
# 19. ACCURACY DIFFERENCE
# ============================================================

accuracy_difference = (

    abs(
        unlearned_accuracy
        -
        retrained_accuracy
    )

)


f1_difference = (

    abs(
        unlearned_f1
        -
        retrained_f1
    )

)


print(
    "\nDifference between "
    "Fine-Tuned and Retrained:"
)


print(
    f"Accuracy difference: "
    f"{accuracy_difference:.4f}"
)


print(
    f"F1 difference: "
    f"{f1_difference:.4f}"
)