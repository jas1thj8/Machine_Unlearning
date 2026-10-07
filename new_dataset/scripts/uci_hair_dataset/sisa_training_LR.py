import pandas as pd
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"
models_folder = folder / "models"

models = []

print("SISA LOGISTIC REGRESSION TRAINING")
print("-" * 40)

for i in range(1, 4):

    shard = pd.read_parquet(
        data_folder / f"sisa_shard_{i}.parquet"
    )

    X = shard.drop(columns=["target", "subject"])
    y = shard["target"]

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    start = time.time()

    model.fit(X, y)

    train_time = time.time() - start

    models.append(model)

    train_pred = model.predict(X)
    accuracy = accuracy_score(y, train_pred)

    print(f"Shard {i}")
    print(f"Samples       : {len(shard)}")
    print(f"Training time : {train_time:.2f} sec")
    print(f"Train accuracy: {accuracy:.4f}")
    print()

print("SISA SHARD TRAINING COMPLETED")