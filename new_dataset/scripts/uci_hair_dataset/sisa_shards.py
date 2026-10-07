import pandas as pd
from pathlib import Path

folder = Path(__file__).parent.parent.parent
data_folder = folder / "data"

train = pd.read_parquet(
    data_folder / "uci_har_train.parquet"
)

NUM_SHARDS = 3

subjects = sorted(train["subject"].unique())

shard_subjects = [
    subjects[i::NUM_SHARDS]
    for i in range(NUM_SHARDS)
]

print("SISA SHARDING")
print("-" * 30)

for i, subject_list in enumerate(shard_subjects, start=1):

    shard = train[train["subject"].isin(subject_list)]

    path = data_folder / f"sisa_shard_{i}.parquet"
    shard.to_parquet(path, index=False)

    print(f"Shard {i}")
    print(f"Subjects : {subject_list}")
    print(f"Samples  : {len(shard)}")
    print()