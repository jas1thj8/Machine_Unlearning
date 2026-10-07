import pandas as pd
import time
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

folder = Path(__file__).parent.parent.parent

data = folder / "data"
models = folder / "models"

train = pd.read_parquet(
    data / "sisa_retain_subject_1.parquet"
)

test = pd.read_parquet(
    data / "uci_har_test.parquet"
)

X_train = train.drop(columns=["target", "subject"])
y_train = train["target"]

X_test = test.drop(columns=["target", "subject"])
y_test = test["target"]

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

start = time.time()

model.fit(X_train, y_train)

retrain_time = time.time() - start

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

joblib.dump(
    model,
    models / "full_retrain_subject_1.joblib"
)

print("FULL RETRAINING")
print("-" * 30)
print("Forgot subject  : 1")
print(f"Training samples: {len(train)}")
print(f"Retraining time : {retrain_time:.4f} sec")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")