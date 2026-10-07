# Overall Information — Machine Unlearning Project

## 1. Project Overview

The project focuses on **Machine Unlearning**, specifically studying different approaches for removing the influence of selected training data from a trained machine learning model.

The main dataset used is the **Wisconsin Breast Cancer Diagnostic Dataset** (`breast-cancer.csv`).

The project compares multiple approaches:

1. Traditional Machine Learning Baseline
2. Pure SISA
3. SISA + Fine-Tuning
4. Pretrained Model + SISA
5. Standalone Fine-Tuning Unlearning
6. Gradient-Based Unlearning
7. Influence-Based Unlearning
8. Full Retraining

The main objective is to compare these approaches in terms of:

* Accuracy
* Precision
* Recall
* F1-score
* Training/unlearning time
* Model parameter changes
* Similarity to full retraining
* Amount of data actually forgotten

---

# 2. Dataset

### Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**

Total samples:

* **569 records**

Original columns:

* `id`
* `diagnosis`
* 30 numerical breast-cancer features
* `Unnamed: 32` may also be present depending on the CSV version

### Target

`diagnosis`

Mapping:

* `B` → `0` — Benign
* `M` → `1` — Malignant

### Features

The dataset contains **30 actual medical feature columns**, including measurements such as:

* radius
* texture
* perimeter
* area
* smoothness
* compactness
* concavity
* concave points
* symmetry
* fractal dimension

Each measurement has corresponding statistical forms such as:

* mean
* standard error
* worst

### Preprocessing

ID-type columns are removed:

* `id`
* `Unnamed: 0`
* `Unnamed: 32`

These are removed during preprocessing because they are identifiers or empty/non-feature columns.

**Important:** These columns are preprocessing removals, not machine-unlearning operations.

---

# 3. Train/Test Split

The dataset is divided using an **80/20 stratified split**.

Approximately:

* Training: **455 samples**
* Testing: **114 samples**

`random_state = 42`

Stratification preserves the Benign/Malignant class distribution between training and testing data.

The test set remains unchanged during the unlearning experiments.

---

# 4. Feature Scaling

The numerical features are standardized using:

```text
StandardScaler
```

The scaler is fitted on the training data and then applied to the test data.

The purpose is to standardize the feature distributions before training the models.

---

# 5. Traditional Machine Learning Baseline

The baseline experiment trains conventional ML models on the original training data.

Models used across the experiments include:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Gradient Boosting
* XGBoost
* MLP

The baseline provides reference performance before applying machine-unlearning techniques.

---

# 6. Pure SISA

## SISA

SISA stands for:

**Sharded, Isolated, Sliced, and Aggregated**

The training data is divided into multiple shards.

In the main SISA experiments:

```text
Number of shards = 3
Random seed = 42
```

Each shard contains a portion of the training data.

A separate model is trained for each shard.

The predictions from the shard models are aggregated to produce the final prediction.

### Unlearning

For the main SISA experiments:

```text
FORGET_SHARD = 0
FORGET_POSITION = 0
```

Therefore:

> **One complete training sample is selected for forgetting from Shard 0 at position 0.**

Only the affected shard needs to be retrained.

Shards 1 and 2 remain unchanged.

### Important

SISA removes/forgets **training records**, not individual features.

There is no operation such as:

```text
remove radius_mean
remove texture_mean
remove area_mean
```

during unlearning.

Instead, the complete selected training record is excluded.

---

# 7. SISA + Fine-Tuning

This is one of the main hybrid approaches in the project.

## Data Structure

The training data is divided into:

```text
20% → Base/Pretraining data
80% → SISA data
```

The SISA portion is then divided into:

```text
3 shards
```

The base model is trained using the 20% base dataset.

The SISA child models are initialized using the base model.

Each child model is then trained/fine-tuned using its corresponding shard.

### Structure

```text
                    Training Data
                         │
             ┌───────────┴───────────┐
             │                       │
        20% Base Data            80% SISA Data
             │                       │
       Base Model              ┌─────┼─────┐
                               │     │     │
                            Shard 0 Shard 1 Shard 2
                               │     │     │
                               ▼     ▼     ▼
                            Child 0 Child 1 Child 2
                               │     │     │
                               └─────┼─────┘
                                     │
                              Probability
                               Aggregation
                                     │
                                     ▼
                              Final Prediction
```

---

# 8. SISA + Fine-Tuning Unlearning

The latest SISA + Fine-Tuning experiment uses:

```text
NUM_SHARDS = 3
FORGET_SHARD = 0
FORGET_POSITION = 0
```

The forgotten sample is therefore:

> **The first training sample in SISA Shard 0.**

The complete training record is removed from the affected shard.

The affected shard is then updated and trained again.

### What exactly is removed?

For the selected record:

* All **30 feature values** belonging to that record
* The corresponding **diagnosis label**

are excluded from the affected shard after the forget operation.

Therefore:

```text
Removed complete training records = 1
Removed feature values = 30
Removed target label = 1
Removed feature columns = 0
```

### What is NOT removed?

The following remain unchanged:

* The feature columns themselves
* Shard 1
* Shard 2
* The test dataset
* Other training records
* The remaining records in Shard 0

### Important distinction

The project is performing:

> **Sample-level / record-level machine unlearning**

It is **not performing feature-level unlearning**.

For example, it does **not** remove the entire:

```text
radius_mean
texture_mean
perimeter_mean
area_mean
```

columns.

Instead, it removes the values of those features **only for the selected forgotten record**.

---

# 9. Exact Forgotten Record

The latest SISA + Fine-Tuning experiment deterministically selects:

```text
Shard = 0
Position = 0
Seed = 42
```

The exact original CSV row/ID and its numerical feature values cannot be determined from the Python script alone because the actual contents of `breast-cancer.csv` are not embedded inside the script.

Once the actual CSV file is available, the exact forgotten record can be reproduced using the same:

```text
random_state = 42
80/20 stratified train/test split
20/80 base/SISA split
3-way stratified SISA shard split
FORGET_SHARD = 0
FORGET_POSITION = 0
```

---

# 10. Pretrained Model + SISA

Another experiment uses a pretrained neural network model followed by SISA.

The model is first trained on the **entire training dataset**.

Configuration:

```text
Pretraining epochs = 60
SISA epochs = 80
Number of shards = 3
Batch size = 16
Learning rate = 0.001
```

After pretraining:

```text
Pretrained Model
       │
       ├── copy → Shard 0 model
       ├── copy → Shard 1 model
       └── copy → Shard 2 model
```

Each child model is then trained on its respective shard.

### Important

This particular implementation does **not yet implement unlearning**.

It does not contain:

* Forget set
* Forgotten sample
* Affected-shard detection
* Sample removal
* Unlearning retraining
* Full-retraining comparison

Therefore, it should currently be considered:

> **Pretrained Model + SISA classification foundation**

rather than a complete unlearning experiment.

---

# 11. Standalone Fine-Tuning Unlearning

This experiment uses a single neural network.

The original model is trained using all 455 training samples.

Then:

```text
10% of training samples → Forget Set
90% → Retain Set
```

Approximately:

```text
Forget = 45 samples
Retain = 410 samples
```

The original model is copied and fine-tuned using the retained samples.

A separate model is fully retrained from scratch using the retained samples.

The results are compared between:

* Original model
* Fine-tuned unlearned model
* Fully retrained model

### Important

This is **not the same as SISA + Fine-Tuning**.

Standalone fine-tuning operates on one global model.

SISA + Fine-Tuning uses a base model and multiple shard-specific child models.

---

# 12. Gradient-Based Unlearning

The Gradient-Based Unlearning experiment uses a PyTorch MLP.

Architecture:

```text
Input
  ↓
64 neurons
  ↓
ReLU
  ↓
32 neurons
  ↓
ReLU
  ↓
1 output
```

The original model is trained on all 455 training samples.

Then:

```text
Forget Ratio = 10%
```

Approximately:

```text
45 forgotten samples
410 retained samples
```

The loss is calculated using the forget set.

A gradient-based update is then applied in the forgetting direction.

The experiment compares:

* Original model
* Gradient-unlearned model
* Fully retrained model

### Important

This method does not use SISA.

It also does not remove feature columns.

It operates on selected **training samples**.

---

# 13. Influence-Based Unlearning

Influence-based unlearning also uses the PyTorch MLP.

Configuration includes:

```text
Forget Ratio = 10%
Learning Rate = 0.001
Damping = 0.01
CG Iterations = 50
```

The method:

1. Calculates gradients associated with the forget set.
2. Uses Hessian-vector products.
3. Uses conjugate-gradient approximation.
4. Estimates an influence vector.
5. Updates the model parameters using the calculated influence.

The resulting model is compared with:

* Original model
* Influence-unlearned model
* Fully retrained model

Additional measurements include parameter changes and distance from the fully retrained model.

### Important

This is:

> **Influence-Based Machine Unlearning**

It is not SISA.

It does not remove individual feature columns.

---

# 14. Full Retraining

Full retraining is used as the reference method for several unlearning experiments.

After removing the forget set:

```text
Original training data
        ↓
Remove forgotten samples
        ↓
Retained dataset
        ↓
Train a completely new model
```

For the 10% forget experiments:

```text
Original training samples = 455
Forgotten samples = 45
Remaining samples = 410
```

Full retraining provides a reference for determining how closely the unlearned model approaches a model that was never trained on the forgotten data.

---

# 15. Comparison of Unlearning Methods

| Method                 | Model Type   |                   Forget Data | SISA | Fine-Tuning | Gradient | Influence | Full Retraining |
| ---------------------- | ------------ | ----------------------------: | ---- | ----------- | -------- | --------- | --------------- |
| Baseline ML            | Classical ML |                          None | No   | No          | No       | No        | No              |
| Pure SISA              | Classical ML |   1 sample in main experiment | Yes  | No          | No       | No        | No              |
| SISA + Fine-Tuning     | MLP          | 1 sample in latest experiment | Yes  | Yes         | No       | No        | No              |
|Pre-trained + SISA      | MLP          |               Not implemented | Yes  |Pre-training | No       | No        | No              |
| Standalone Fine-Tuning | MLP          |                    45 samples | No   | Yes         | No       | No        | No              |
| Gradient Unlearning    | MLP          |                    45 samples | No   | No          | Yes      | No        | No              |
| Influence Unlearning   | MLP          |                    45 samples | No   | No          | No       | Yes       | No              |
| Full Retraining        | ML/MLP       |           45 samples excluded | No   | No          | No       | No        | Yes             |

---

# 16. Pure SISA Experimental Results

The reported pure-SISA results are:

| Classifier          | Accuracy | Precision | Recall |     F1 |
| ------------------- | -------: | --------: | -----: | -----: |
| Logistic Regression |   98.25% |    98.29% | 98.25% | 98.24% |
| SVM                 |   98.25% |    98.29% | 98.25% | 98.24% |
| XGBoost             |   98.25% |    98.29% | 98.25% | 98.24% |
| Gradient Boosting   |   95.61% |    95.90% | 95.61% | 95.55% |
| Random Forest       |   95.61% |    95.69% | 95.61% | 95.58% |
| KNN                 |   94.74% |    94.88% | 94.74% | 94.68% |
| Decision Tree       |   93.86% |    94.08% | 93.86% | 93.77% |
| Gaussian NB         |   92.11% |    92.11% | 92.11% | 92.04% |

---

# 17. Pure SISA Unlearning Results

| Classifier          | Before |  After |   Change |
| ------------------- | -----: | -----: | -------: |
| Logistic Regression | 98.25% | 98.25% |  0.00 pp |
| Random Forest       | 95.61% | 95.61% |  0.00 pp |
| SVM                 | 98.25% | 97.37% | -0.88 pp |
| Decision Tree       | 93.86% | 93.86% |  0.00 pp |
| KNN                 | 94.74% | 94.74% |  0.00 pp |
| Gaussian NB         | 92.11% | 92.98% | +0.87 pp |
| Gradient Boosting   | 95.61% | 95.61% |  0.00 pp |
| XGBoost             | 98.25% | 98.25% |  0.00 pp |

---

# 17" Table for which is the strongest model

| Method                 |  SISA  | Fine-Tuning  | Selective Unlearning | Hybrid |
| ---------------------- | ----:  | ----------:  | -------------------: | -----: |
| Pure SISA              |     ✅ |           ❌ |                    ✅ |      ❌ |
| Standalone Fine-Tuning |     ❌ |           ✅ |                    ❌ |      ❌ |
| Gradient Unlearning    |     ❌ |           ❌ |                    ✅ |      ❌ |
| Influence Unlearning   |     ❌ |           ❌ |                    ✅ |      ❌ |
| Pretrained + SISA      |     ✅ |     Partial  |                    ❌ |      ✅ |
| **SISA + Fine-Tuning** | **✅** |       **✅** |                **✅** |  **✅** |

# 18. Important Methodological Point About SISA

The current implementation is a **shard-level SISA implementation**.

The basic process is:

```text
Training Dataset
      ↓
Create Shards
      ↓
Train Independent Models
      ↓
Aggregate Predictions
      ↓
Forget Selected Sample
      ↓
Retrain Affected Shard
      ↓
Aggregate Again
```

A stricter implementation of original SISA would also use **slices/checkpoints** so that the affected model can be restored to the appropriate state before the forgotten sample was introduced.

Therefore, the current implementation should be described accurately as a **SISA-style shard-based unlearning implementation** unless checkpoint/slicing functionality is added.

---

# 19. Key Difference Between the Main Approaches

### Pure SISA

```text
Shard → Train model → Forget sample → Retrain affected shard
```

### SISA + Fine-Tuning

```text
Base data → Base model
                  ↓
             Initialize
                  ↓
        ┌─────────┼─────────┐
      Shard 0   Shard 1   Shard 2
        ↓         ↓         ↓
     Fine-tune Fine-tune Fine-tune
        ↓         ↓         ↓
             Aggregate
```

### Gradient Unlearning

```text
Original Model
      ↓
Forget-set gradient
      ↓
Gradient update
      ↓
Unlearned Model
```

### Influence Unlearning

```text
Original Model
      ↓
Forget gradients
      ↓
Hessian / CG
      ↓
Influence estimate
      ↓
Parameter update
```

### Full Retraining

```text
Remove forget data
      ↓
Train completely new model
```

---

# 20. Exact Meaning of "Data Removed"

For the latest **SISA + Fine-Tuning** experiment:

```text
Training records removed = 1
Feature columns removed = 0
Feature values removed from the selected record = 30
Target label removed = 1
Shard 0 affected = Yes
Shard 1 affected = No
Shard 2 affected = No
Test data affected = No
```

The important distinction is:

> **The project forgets complete training records, not feature columns.**

If one record contains:

```text
30 features + 1 diagnosis label
```

then forgetting that record means that particular record's:

```text
30 feature values + diagnosis label
```

are excluded from further training of the affected model.

It does **not** mean that any of the 30 feature columns are deleted from the dataset.

---

# 21. Final Project Methodology

The overall methodology can be represented as:

```text
                    Breast Cancer Dataset
                            │
                            ▼
                    Data Preprocessing
                            │
                ┌───────────┴───────────┐
                │                       │
           80% Training             20% Testing
                │
                ▼
        ┌───────────────────┐
        │ Experimental       │
        │ Methods            │
        └───────────────────┘
                │
     ┌──────────┼───────────┬─────────────┐
     │          │           │             │
     ▼          ▼           ▼             ▼
  Baseline    Pure SISA   SISA + FT   Pretrained
                                         + SISA
     │          │           │             │
     └──────────┼───────────┴─────────────┘
                │
                ▼
       Other Unlearning Methods
                │
       ┌────────┼─────────┐
       │        │         │
       ▼        ▼         ▼
   Gradient  Influence  Fine-Tuning
       │        │         │
       └────────┼─────────┘
                │
                ▼
          Full Retraining
          Reference Model
                │
                ▼
       Performance Comparison
                │
       ┌────────┼───────────────┐
       │        │               │
    Accuracy   F1            Time
       │        │               │
       └────────┼───────────────┘
                │
                ▼
       Machine Unlearning
          Evaluation
```

---

# 22. Final Research Focus

The project investigates whether machine learning models can **forget selected training data efficiently without completely retraining the model**.

The main comparison is between:

* **SISA-based unlearning**
* **SISA + Fine-Tuning**
* **Gradient-based unlearning**
* **Influence-based unlearning**
* **Standalone fine-tuning**
* **Full retraining**

The evaluation focuses on the trade-off between:

```text
Forgetting effectiveness
        ↕
Model performance
        ↕
Computational cost
        ↕
Similarity to full retraining
```

The central unlearning operation in the SISA experiments is **sample-level deletion/forgetting**, where selected complete training records are excluded from the affected model's subsequent training.
