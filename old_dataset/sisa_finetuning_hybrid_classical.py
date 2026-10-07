"""SISA + fine-tuning hybrid for classical classifiers.

Each model first learns a small common pre-training set.  A separate copy of
that base model is then adapted with one SISA shard.  To forget a sample, only
the copy belonging to its shard is rebuilt from the unchanged base model.

Examples:
    python sisa_finetuning_hybrid_classical.py --model logistic_regression
    python sisa_finetuning_hybrid_classical.py --model all
"""

import argparse
import copy
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 42
CSV_PATH = r"Y:\VSc\MUL\breast-cancer.csv"
NUM_SHARDS = 3
PRETRAIN_RATIO = 0.20
FORGET_SHARD = 0
FORGET_POSITION = 0


def load_data(path):
    """Load the CSV and return numeric features plus zero-based labels."""
    df = pd.read_csv(path)
    target_candidates = {
        "target", "label", "class", "failure", "battery_failure",
        "failure_label", "status", "diagnosis", "y",
    }
    target = next((c for c in df.columns if c.lower().strip() in target_candidates), None)
    if target is None:
        raise ValueError("Target column not found. Set a known target column in load_data().")

    drop_columns = [
        c for c in df.columns
        if c.lower().strip() in {"id", "unnamed: 0", "unnamed: 32"}
    ]
    df = df.drop(columns=drop_columns)
    features = df.drop(columns=[target]).copy()
    for column in features.columns:
        numeric = pd.to_numeric(features[column], errors="coerce")
        if numeric.notna().all():
            features[column] = numeric
        else:
            features[column] = LabelEncoder().fit_transform(features[column].astype(str))

    valid = features.notna().all(axis=1) & df[target].notna()
    X = features.loc[valid].to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(df.loc[valid, target].astype(str))
    return X, y, target


def stratified_shards(y, count):
    """Return balanced, reproducible index lists for the SISA stage."""
    rng = np.random.default_rng(SEED)
    shards = [[] for _ in range(count)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for shard, split in zip(shards, np.array_split(indices, count)):
            shard.extend(split.tolist())
    for shard in shards:
        rng.shuffle(shard)
    return shards


def make_model(name):
    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, warm_start=True, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=150, warm_start=True, n_jobs=-1, random_state=SEED
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
    }
    return models[name]


def adapt_from_base(name, base_model, X_base, y_base, X_shard, y_shard):
    """Fine-tune a base model with one shard.

    LR and RF retain fitted state through warm_start.  SVC, decision trees and
    KNN do not expose incremental fitting in scikit-learn, so their equivalent
    adaptation is a deterministic fit on base + shard data.  GaussianNB is
    genuinely updated with partial_fit.
    """
    model = copy.deepcopy(base_model)
    if name == "random_forest":
        model.n_estimators += 75
        model.fit(X_shard, y_shard)
    elif name == "naive_bayes":
        model.partial_fit(X_shard, y_shard, classes=np.unique(y_base))
    elif name == "logistic_regression":
        model.fit(X_shard, y_shard)
    else:
        model = clone(base_model)
        model.fit(np.vstack((X_base, X_shard)), np.concatenate((y_base, y_shard)))
    return model


def aggregate(models, X):
    probabilities = np.mean([model.predict_proba(X) for model in models], axis=0)
    return probabilities.argmax(axis=1)


def score(title, models, X_test, y_test):
    predictions = aggregate(models, X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    print(f"{title}: accuracy={accuracy:.4f}, weighted F1={f1:.4f}")
    return accuracy, f1


def run_experiment(name):
    X, y, target = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_base, X_sisa, y_base, y_sisa = train_test_split(
        X_train, y_train, test_size=1 - PRETRAIN_RATIO, random_state=SEED, stratify=y_train
    )
    shard_indices = stratified_shards(y_sisa, NUM_SHARDS)

    print("\n" + "=" * 70)
    print(f"SISA + FINE-TUNING HYBRID | {name.upper()}")
    print(f"Target: {target}; base samples: {len(y_base)}; SISA samples: {len(y_sisa)}")

    base_model = make_model(name)
    start = time.perf_counter()
    base_model.fit(X_base, y_base)
    base_time = time.perf_counter() - start

    shard_models = []
    shard_time = 0.0
    for shard_id, indices in enumerate(shard_indices):
        start = time.perf_counter()
        model = adapt_from_base(name, base_model, X_base, y_base, X_sisa[indices], y_sisa[indices])
        elapsed = time.perf_counter() - start
        shard_models.append(model)
        shard_time += elapsed
        print(f"Shard {shard_id}: {len(indices)} samples, adaptation time={elapsed:.3f}s")

    original = score("Before unlearning", shard_models, X_test, y_test)

    affected = FORGET_SHARD
    indices = list(shard_indices[affected])
    forgotten = indices.pop(FORGET_POSITION)
    start = time.perf_counter()
    shard_models[affected] = adapt_from_base(
        name, base_model, X_base, y_base, X_sisa[indices], y_sisa[indices]
    )
    unlearning_time = time.perf_counter() - start
    updated = score("After unlearning ", shard_models, X_test, y_test)

    print(f"Forgot SISA-row {forgotten} from shard {affected}")
    print(f"Base training={base_time:.3f}s; all shard adaptation={shard_time:.3f}s")
    print(f"Affected-shard unlearning={unlearning_time:.3f}s")
    return {"model": name, "before_accuracy": original[0], "after_accuracy": updated[0], "unlearning_seconds": unlearning_time}


def main():
    available = ["logistic_regression", "random_forest", "svm", "decision_tree", "knn", "naive_bayes"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*available, "all"], default="all")
    args = parser.parse_args()
    selected = available if args.model == "all" else [args.model]
    results = [run_experiment(name) for name in selected]
    print("\nSummary")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
