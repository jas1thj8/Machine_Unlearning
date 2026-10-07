import pandas as pd
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"


train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

test = pd.read_parquet(
    data_folder / "uci_har_test.parquet"
)


X_train = train.drop(columns=["target", "subject"])
y_train = train["target"]

X_test = test.drop(columns=["target", "subject"])
y_test = test["target"]


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


print("Training Random Forest...")

start = time.time()

model.fit(X_train, y_train)

train_time = time.time() - start


print("Testing...")

start = time.time()

y_pred = model.predict(X_test)

test_time = time.time() - start


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print()
print("Random Forest")
print("-------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Train Time: {train_time:.2f} sec")
print(f"Test Time : {test_time:.2f} sec")