#SISA algorithm
#SHARED,ISOLATED,SLICED,AGGREGATED algorithm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Y:\\VSc\\MUL\\breast-cancer.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Remove unnecessary columns if they exist
columns_to_drop = []

for column in ["id", "ID", "Unnamed: 32"]:
    if column in df.columns:
        columns_to_drop.append(column)

df = df.drop(columns=columns_to_drop)


# ============================================================
# 3. IDENTIFY TARGET COLUMN
# ============================================================

# Kaggle versions commonly use "diagnosis"
if "diagnosis" in df.columns:

    # B = Benign -> 0
    # M = Malignant -> 1

    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

else:

    # If your dataset uses another target name,
    # change this manually.
    print("Target column not found.")
    print("Available columns:", df.columns.tolist())
    raise ValueError("Please set the correct target column.")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
}


# ============================================================
# 6. TRAIN MODELS
# ============================================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Scaling is important for:
    # Logistic Regression
    # SVM
    # KNN
    # MLP

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Store trained model
    trained_models[name] = pipeline

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_model_name = results_df["F1 Score"].idxmax()

print("\nBest model:", best_model_name)
print(
    "Best F1 Score:",
    results_df.loc[best_model_name, "F1 Score"]
)