import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

DATASET = "Y:\\VSc\\MUL\\breast-cancer.csv"
RANDOM_STATE = 42
FORGET_RATIO = 0.10
EPOCHS = 100
LEARNING_RATE = 0.001
DAMPING = 0.01
CG_ITERATIONS = 50

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

df = pd.read_csv(DATASET)
print("Dataset shape:", df.shape)

columns_to_remove = []
for column in ["id", "ID", "Unnamed: 32"]:
    if column in df.columns:
        columns_to_remove.append(column)
df = df.drop(columns=columns_to_remove)

if "diagnosis" not in df.columns:
    raise ValueError("Column 'diagnosis' was not found.")

df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

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
criterion = nn.BCEWithLogitsLoss()

def train_model(model, X, y, epochs=100):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    model.train()
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return time.time() - start

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()
    accuracy = accuracy_score(
        y.numpy(),
        predictions.numpy()
    )
    f1 = f1_score(
        y.numpy(),
        predictions.numpy()
    )
    return accuracy, f1

print("\n" + "=" * 60)
print("1. TRAINING ORIGINAL MODEL")
print("=" * 60)

original_model = CancerMLP(input_size)

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

print(f"Training time: {original_training_time:.4f} sec")
print(f"Accuracy: {original_accuracy:.4f}")
print(f"F1 Score: {original_f1:.4f}")

number_of_forget_samples = int(
    len(X_train) * FORGET_RATIO
)

rng = np.random.default_rng(RANDOM_STATE)

forget_indices = rng.choice(
    len(X_train),
    size=number_of_forget_samples,
    replace=False
)

forget_indices = np.sort(forget_indices)

print("\n" + "=" * 60)
print("2. CREATING FORGET SET")
print("=" * 60)
print("Forget samples:", len(forget_indices))

retain_mask = np.ones(
    len(X_train),
    dtype=bool
)

retain_mask[forget_indices] = False

X_retain = X_train[retain_mask]
y_retain = y_train[retain_mask]
X_forget = X_train[forget_indices]
y_forget = y_train[forget_indices]

print("Retain samples:", len(X_retain))
print("Forget samples:", len(X_forget))

def get_parameters(model):
    return torch.cat([
        parameter.detach().flatten()
        for parameter in model.parameters()
    ])

def set_parameters(model, vector):
    position = 0
    for parameter in model.parameters():
        number = parameter.numel()
        parameter.data.copy_(
            vector[
                position:position + number
            ].reshape(parameter.shape)
        )
        position += number

def get_grad_vector(model):
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            gradients.append(
                torch.zeros_like(parameter).flatten()
            )
        else:
            gradients.append(parameter.grad.flatten())
    return torch.cat(gradients)

def hessian_vector_product(model, X, y, vector):
    model.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    gradients = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=True
    )
    gradient_vector = torch.cat([
        gradient.flatten()
        for gradient in gradients
    ])
    gradient_dot_vector = torch.sum(
        gradient_vector * vector
    )
    second_gradients = torch.autograd.grad(
        gradient_dot_vector,
        model.parameters(),
        retain_graph=False
    )
    hvp = torch.cat([
        gradient.flatten()
        for gradient in second_gradients
    ])
    return hvp

def conjugate_gradient(
    model,
    X,
    y,
    b,
    damping=0.01,
    iterations=50
):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(iterations):
        Hp = hessian_vector_product(
            model,
            X,
            y,
            p
        )
        Hp = Hp + damping * p
        denominator = torch.dot(p, Hp)

        if abs(denominator.item()) < 1e-12:
            break

        alpha = rs_old / denominator
        x = x + alpha * p
        r = r - alpha * Hp
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < 1e-6:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

print("\n" + "=" * 60)
print("3. CALCULATING FORGET-SET INFLUENCE")
print("=" * 60)

original_model.zero_grad()

forget_outputs = original_model(X_forget)

forget_loss = criterion(
    forget_outputs,
    y_forget
)

forget_gradients = torch.autograd.grad(
    forget_loss,
    original_model.parameters(),
    create_graph=False
)

forget_gradient_vector = torch.cat([
    gradient.flatten()
    for gradient in forget_gradients
])

print("Forget-set loss:", forget_loss.item())
print("Gradient dimension:", len(forget_gradient_vector))

print("\nCalculating approximate inverse-Hessian product...")

start_time = time.time()

influence_vector = conjugate_gradient(
    original_model,
    X_train,
    y_train,
    forget_gradient_vector,
    damping=DAMPING,
    iterations=CG_ITERATIONS
)

influence_time = time.time() - start_time

print(
    "Influence calculation time:",
    f"{influence_time:.4f} sec"
)

print("\n" + "=" * 60)
print("4. APPLYING INFLUENCE UPDATE")
print("=" * 60)

unlearned_model = copy.deepcopy(original_model)

original_parameters = get_parameters(
    original_model
)

updated_parameters = (
    original_parameters + influence_vector
)

set_parameters(
    unlearned_model,
    updated_parameters
)

unlearned_accuracy, unlearned_f1 = evaluate_model(
    unlearned_model,
    X_test,
    y_test
)

print(f"Accuracy: {unlearned_accuracy:.4f}")
print(f"F1 Score: {unlearned_f1:.4f}")

print("\n" + "=" * 60)
print("5. FULL RETRAINING REFERENCE")
print("=" * 60)

retrained_model = CancerMLP(input_size)

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

print(f"Retraining time: {retraining_time:.4f} sec")
print(f"Accuracy: {retrained_accuracy:.4f}")
print(f"F1 Score: {retrained_f1:.4f}")

original_parameters = get_parameters(
    original_model
)

unlearned_parameters = get_parameters(
    unlearned_model
)

retrained_parameters = get_parameters(
    retrained_model
)

influence_parameter_change = torch.norm(
    original_parameters - unlearned_parameters
).item()

difference_from_retrained = torch.norm(
    unlearned_parameters - retrained_parameters
).item()

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Original Model",
        "Influence Unlearned",
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
    "Time (sec)": [
        original_training_time,
        influence_time,
        retraining_time
    ]
})

print(
    results.to_string(index=False)
)

print("\n" + "=" * 60)
print("INFLUENCE ANALYSIS")
print("=" * 60)

print(
    "Parameter change from original:",
    influence_parameter_change
)

print(
    "Distance from fully retrained model:",
    difference_from_retrained
)

if influence_time > 0:
    speedup = retraining_time / influence_time
    print(
        "Approximate speedup:",
        f"{speedup:.2f}x"
    )

accuracy_difference = abs(
    unlearned_accuracy - retrained_accuracy
)

f1_difference = abs(
    unlearned_f1 - retrained_f1
)

print(
    "\nAccuracy difference:",
    f"{accuracy_difference:.4f}"
)

print(
    "F1 difference:",
    f"{f1_difference:.4f}"
)