import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_PATH = "data.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("WISCONSIN BREAST CANCER - BASELINE MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv("MUL\\breast-cancer.csv")

print("\nDataset shape:", df.shape)


# ============================================================
# 3. PREPROCESSING
# ============================================================

# Remove unnecessary ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Remove unnamed columns sometimes created by CSV exports
unnamed_columns = [
    col for col in df.columns
    if col.lower().startswith("unnamed")
]

if unnamed_columns:
    df = df.drop(columns=unnamed_columns)


# Diagnosis is the target
if "diagnosis" not in df.columns:
    raise ValueError(
        "The dataset must contain a 'diagnosis' column."
    )

# Convert diagnosis:
# M = 1 (Malignant)
# B = 0 (Benign)

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Remove rows with invalid target values
df = df.dropna(subset=["diagnosis"])


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].astype(int)


# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values with median
X = X.fillna(X.median())


print("Final dataset shape:", X.shape)

print("\nClass distribution:")
print(
    y.value_counts()
    .rename(index={
        0: "Benign",
        1: "Malignant"
    })
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. DEFINE ALL MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # 1. Logistic Regression
    # --------------------------------------------------------
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]),

    # --------------------------------------------------------
    # 2. Decision Tree
    # --------------------------------------------------------
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    # --------------------------------------------------------
    # 3. Random Forest
    # --------------------------------------------------------
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    # --------------------------------------------------------
    # 4. SVM
    # --------------------------------------------------------
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        ))
    ]),

    # --------------------------------------------------------
    # 5. KNN
    # --------------------------------------------------------
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    # --------------------------------------------------------
    # 6. Gaussian Naive Bayes
    # --------------------------------------------------------
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    # --------------------------------------------------------
    # 7. Gradient Boosting
    # --------------------------------------------------------
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    ),

    # --------------------------------------------------------
    # 8. XGBoost
    # --------------------------------------------------------
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 6. TRAIN + EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    start_time = time.perf_counter()

    # Train
    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Print
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nTraining time: {training_time:.4f} seconds")

    # Save result
    results.append({
        "Algorithm": name,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 (%)": f1 * 100,
        "Training Time (s)": training_time
    })


# ============================================================
# 7. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy (%)",
    ascending=False
).reset_index(drop=True)


print("\n\n")
print("=" * 90)
print("FINAL BASELINE RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy (%)": "{:.2f}".format,
            "Precision (%)": "{:.2f}".format,
            "Recall (%)": "{:.2f}".format,
            "F1 (%)": "{:.2f}".format,
            "Training Time (s)": "{:.4f}".format
        }
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("baseline_results.csv")

print("\nBaseline experiment completed.")