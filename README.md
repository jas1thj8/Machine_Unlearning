# Machine Unlearning: SISA and Fine-Tuning Hybrids

This project evaluates machine-unlearning methods on the **Wisconsin Breast Cancer Diagnostic dataset**. It compares standard **SISA (Sharded, Isolated, Sliced, and Aggregated)** classifiers with **SISA + Fine-Tuning hybrid** classifiers, while investigating how different learning algorithms affect predictive performance and unlearning cost.

The project has now been expanded to include **gradient-based unlearning, influence-based unlearning, standalone fine-tuning unlearning, pretrained-model + SISA, and full retraining** as additional reference methods.

---

## Methods

### Original SISA

The training data is divided into independent stratified shards. One classifier is trained for each shard, and predictions from the shard models are aggregated to produce the final prediction.

For sample-level unlearning, the training sample is located in its corresponding shard. That sample is removed, and only the affected shard model is retrained or updated while the unaffected shard models are retained.

The current pure-SISA implementations include:

* Logistic Regression: `sisaLR.py`
* Random Forest: `sisaRF.py`
* Support Vector Machine: `sisaSVM.py`
* Decision Tree: `sisaDT.py`
* K-Nearest Neighbors: `sisaKNN.py`
* Gaussian Naive Bayes: `sisaNB.py`
* Gradient Boosting: `sisaGDB.py`
* XGBoost: `sisaXGBoost.py`

The current experiments use the Wisconsin breast-cancer dataset containing **569 samples and 30 diagnostic features**, with the dataset split into training and testing sets.

### Important SISA Terminology

The current implementation performs **shard-level sample unlearning**.

It should not be described as a complete checkpoint-based SISA implementation because the current version does not yet implement the full slicing and checkpoint mechanism.

The current process is:

```text
Training Data
     ↓
Stratified Sharding
     ↓
Independent Shard Models
     ↓
Prediction Aggregation
     ↓
Forget Sample
     ↓
Remove Sample from Affected Shard
     ↓
Retrain/Update Affected Shard
     ↓
Aggregate Again
```

---

# SISA + Fine-Tuning Hybrid

The main hybrid approach combines a shared base model with SISA-style shard-specific models.

The training data is divided into:

```text
20% → Base/Pretraining Data
80% → SISA Data
```

The 20% base dataset is used to train a shared base model.

The remaining 80% is divided into three stratified SISA shards.

Each shard model is initialized from the shared base model and adapted using its corresponding shard.

```text
                 Training Data
                       │
             ┌─────────┴─────────┐
             │                   │
        20% Base Data       80% SISA Data
             │                   │
             ▼                   ▼
        Base Model          ┌────┼────┐
                            │    │    │
                         Shard0 Shard1 Shard2
                            │    │    │
                            ▼    ▼    ▼
                         Fine-Tuned Models
                            │    │    │
                            └────┼────┘
                                 ▼
                         Prediction Aggregation
                                 │
                                 ▼
                           Final Prediction
```

For unlearning, the affected child model is updated using the remaining data in its shard while the unaffected child models remain unchanged.

### Current Hybrid Implementation

Run all hybrid models:

```powershell
python sisa_finetuning_hybrid_classical.py --model all
```

Run a single hybrid model:

```powershell
python sisa_finetuning_hybrid_classical.py --model decision_tree
```

---

# Exact Data Forgotten in the Current SISA + Fine-Tuning Experiment

The latest implementation uses:

```text
NUM_SHARDS = 3
FORGET_SHARD = 0
FORGET_POSITION = 0
```

Therefore:

> **One complete training record from SISA Shard 0, position 0, is selected for forgetting.**

The forgotten record contains:

```text
30 feature values
+
1 diagnosis label
```

The operation therefore removes the selected **training record**, not an entire feature column.

| Item                                       | Removed/Changed    |
| ------------------------------------------ | ------------------ |
| Complete selected training record          | **Yes — 1 record** |
| 30 feature values belonging to that record | **Yes**            |
| Diagnosis label of that record             | **Yes**            |
| Feature columns themselves                 | **No**             |
| Shard 0                                    | **Affected**       |
| Shard 1                                    | **Unchanged**      |
| Shard 2                                    | **Unchanged**      |
| Test data                                  | **Unchanged**      |

The exact CSV ID and numerical values of the forgotten record require the actual `breast-cancer.csv` file and the deterministic split procedure. They should not be inferred from the script alone.

---

# Additional Unlearning Methods

## Pretrained Model + SISA

A separate experiment first trains a neural-network model on the complete training split.

The pretrained model is then copied to multiple SISA child models.

```text
Full Training Data
        ↓
Pretrained Model
        ↓
   ┌────┼────┐
   ↓    ↓    ↓
 S0     S1    S2
   ↓    ↓    ↓
Child Models
   └────┼────┘
        ↓
   Aggregation
```

The current implementation uses:

* 3 SISA shards
* 60 pretraining epochs
* 80 SISA training epochs
* Batch size = 16
* Learning rate = 0.001

### Current Status

This implementation is currently a **Pretrained Model + SISA classification foundation**.

It does not yet contain a complete forgetting operation.

Therefore, it should not be counted as a completed unlearning experiment until a forget set, affected-shard update, and post-unlearning evaluation are added.

---

# Standalone Fine-Tuning Unlearning

This experiment uses a single global neural-network model rather than SISA.

The training set contains approximately:

```text
455 training samples
```

The forget ratio is:

```text
10%
```

Therefore:

```text
45 samples → Forget Set
410 samples → Retain Set
```

The original model is trained using all training samples.

A copy is then fine-tuned using only the retained samples.

The result is compared with a model fully retrained from scratch using the retained data.

This provides a reference for evaluating how closely fine-tuning approaches complete retraining.

---

# Gradient-Based Unlearning

The gradient-based experiment uses a PyTorch MLP.

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

Approximately 10% of the training data is selected as the forget set:

```text
45 samples → Forget
410 samples → Retain
```

The forget-set loss is used to calculate gradients.

A gradient-based parameter update is then applied in the forgetting direction.

The resulting model is compared with:

* Original model
* Gradient-unlearned model
* Fully retrained model

This method does not use SISA.

---

# Influence-Based Unlearning

The influence-based experiment also uses the PyTorch MLP.

The method calculates the influence of the forgotten samples on the trained model using:

* Forget-set gradients
* Hessian-vector products
* Conjugate-gradient approximation
* Damping

The configuration includes:

```text
Forget Ratio = 10%
Damping = 0.01
CG Iterations = 50
```

The influence vector is used to modify the original model parameters.

The result is compared with:

* Original model
* Influence-unlearned model
* Fully retrained model

This method does not use SISA or feature-level deletion.

---

# Full Retraining

Full retraining is used as the primary reference for unlearning.

After removing the forget set:

```text
455 original training samples
        ↓
45 forgotten samples
        ↓
410 retained samples
        ↓
Train a new model from scratch
```

The fully retrained model represents the expected model behavior when the forgotten data had never been included in training.

This allows the project to compare approximate unlearning methods against a complete retraining reference.

---

# Original SISA Results

The latest experiments provide the following initial held-out test results:

| Classifier           |   Accuracy | Precision | Recall |   F1 Score |
| -------------------- | ---------: | --------: | -----: | ---------: |
| Logistic Regression  | **98.25%** |    98.29% | 98.25% | **98.24%** |
| SVM                  | **98.25%** |    98.29% | 98.25% | **98.24%** |
| XGBoost              | **98.25%** |    98.29% | 98.25% | **98.24%** |
| Gradient Boosting    |     95.61% |    95.90% | 95.61% |     95.55% |
| Random Forest        |     95.61% |    95.69% | 95.61% |     95.58% |
| KNN                  |     94.74% |    94.88% | 94.74% |     94.68% |
| Decision Tree        |     93.86% |    94.08% | 93.86% |     93.77% |
| Gaussian Naive Bayes |     92.11% |    92.11% | 92.11% |     92.04% |

The highest observed pure-SISA accuracy is **98.25%**, achieved by Logistic Regression, SVM, and XGBoost.

---

# SISA Unlearning Results

| Classifier           | Before |      After |       Change | Unlearning Time |
| -------------------- | -----: | ---------: | -----------: | --------------: |
| Logistic Regression  | 98.25% |     98.25% |      0.00 pp |          0.01 s |
| Random Forest        | 95.61% |     95.61% |      0.00 pp |          0.21 s |
| SVM                  | 98.25% |     97.37% | **-0.88 pp** |         ~0.00 s |
| Decision Tree        | 93.86% |     93.86% |      0.00 pp |         ~0.00 s |
| KNN                  | 94.74% |     94.74% |      0.00 pp |         ~0.00 s |
| Gaussian Naive Bayes | 92.11% | **92.98%** | **+0.87 pp** |         ~0.00 s |
| Gradient Boosting    | 95.61% |     95.61% |      0.00 pp |          0.13 s |
| XGBoost              | 98.25% |     98.25% |      0.00 pp |          0.05 s |

These results show that the selected forgotten sample has different effects depending on the learning algorithm.

---

# SISA vs SISA + Fine-Tuning

The previously obtained comparison is:

| Classifier           | Original SISA | SISA + Fine-Tuning |   Difference |
| -------------------- | ------------: | -----------------: | -----------: |
| Logistic Regression  |        98.25% |         **98.25%** |      0.00 pp |
| Random Forest        |        95.61% |             94.74% |     -0.88 pp |
| SVM                  |        98.25% |             96.49% |     -1.75 pp |
| Decision Tree        |        93.86% |         **96.49%** | **+2.63 pp** |
| KNN                  |        94.74% |             93.86% |     -0.88 pp |
| Gaussian Naive Bayes |        92.11% |             92.11% |      0.00 pp |

### Interpretation

The hybrid does **not** consistently increase predictive accuracy across all classifiers.

The observed changes are:

* Logistic Regression: unchanged
* Gaussian Naive Bayes: unchanged
* Decision Tree: +2.63 percentage points
* Random Forest: -0.88 percentage points
* KNN: -0.88 percentage points
* SVM: -1.75 percentage points

Therefore, the current results support the conclusion that the effect of the hybrid approach is **algorithm-dependent**.

---

# Main Hybrid Model for the Project

Among the implemented approaches, the **SISA + Fine-Tuning Hybrid** is the project's main hybrid architecture because it combines:

```text
Shared Base Training
        +
SISA Sharding
        +
Shard-Specific Fine-Tuning
        +
Selective Sample Unlearning
        +
Prediction Aggregation
```

The architecture can be summarized as:

```text
                  Training Dataset
                         │
                  ┌──────┴──────┐
                  │             │
               Base 20%      SISA 80%
                  │             │
                  ▼             ▼
             Base Model    ┌────┼────┐
                           │    │    │
                         S0    S1    S2
                           │    │    │
                           ▼    ▼    ▼
                       Fine-Tuning
                           │
                           ▼
                       Aggregation
                           │
                           ▼
                       Final Model
                           │
                     Forget Request
                           │
                           ▼
                    Affected Shard
                           │
                     Remove Record
                           │
                           ▼
                  Update/Fine-Tune Shard
                           │
                           ▼
                    New Aggregation
```

For the current project, this should be described as the:

> **Primary proposed hybrid architecture**

rather than claiming that it is universally the highest-performing unlearning method.

The current reported hybrid accuracy is **98.25% for Logistic Regression**, while the largest improvement over the corresponding pure-SISA result is **+2.63 percentage points for Decision Tree**.

---

# Current Findings

* The highest observed pure-SISA accuracy is **98.25%**, achieved by Logistic Regression, SVM, and XGBoost.
* The highest reported SISA + Fine-Tuning hybrid accuracy is **98.25%**, achieved by Logistic Regression.
* The largest observed hybrid improvement is **+2.63 percentage points for Decision Tree**.
* Logistic Regression and Gaussian Naive Bayes retain the same reported accuracy in the current SISA vs hybrid comparison.
* Random Forest, SVM, and KNN show lower reported accuracy with the hybrid configuration.
* Pure-SISA unlearning generally produces little or no change in the current held-out accuracy measurements.
* SVM shows a **0.88 percentage-point decrease** after the selected sample is forgotten.
* Gaussian Naive Bayes shows a **0.87 percentage-point increase** after the selected sample is forgotten.
* The effect of forgetting depends on the learning algorithm and the selected training sample.
* The current SISA + Fine-Tuning implementation performs **sample-level unlearning**, not feature-level unlearning.
* The selected forgotten record contains 30 feature values and one diagnosis label.
* **No diagnostic feature column is removed during the unlearning operation.**

---

# Important Methodological Update

The current implementation should be distinguished from the **full checkpoint-based SISA architecture**.

The current experiments perform:

> **Shard-level sample unlearning**

The affected shard is updated after the forgotten sample is removed.

A complete SISA implementation additionally uses:

> **Sharding → Isolation → Slicing → Checkpointing → Aggregation**

With checkpoint-based SISA, the affected shard can be restored to an appropriate checkpoint and retrained from that point rather than rebuilding the entire shard from scratch.

Therefore, the current implementation should be described as:

> **Pure SISA baseline with shard-level sample unlearning**

and the planned enhanced implementation as:

> **Sliced and checkpoint-based SISA for selective retraining**

---

# Current Research Structure

The project is now organized into the following layers.

## Layer 1 — Conventional Baseline

Train conventional full-data models to establish reference predictive performance and training cost.

---

## Layer 2 — Pure SISA

Train the same learning algorithms independently across SISA shards.

Current algorithms:

**Logistic Regression → Random Forest → SVM → Decision Tree → KNN → Gaussian NB → Gradient Boosting → XGBoost**

Evaluate:

* Accuracy
* Precision
* Recall
* F1
* Training time
* Unlearning time

---

## Layer 3 — SISA + Fine-Tuning Hybrid

Use a shared base model and adapt copies of it to individual SISA shards.

Evaluate the hybrid against corresponding pure-SISA models.

The primary hybrid structure is:

**Base Training → SISA Sharding → Shard Fine-Tuning → Aggregation → Selective Unlearning**

---

## Layer 4 — Additional Unlearning Methods

Compare the SISA approaches with:

* Standalone Fine-Tuning
* Gradient-Based Unlearning
* Influence-Based Unlearning
* Full Retraining

This provides multiple approaches for evaluating the trade-off between model utility and unlearning cost.

---

## Layer 5 — Full SISA

Implement:

**Sharding → Isolation → Slicing → Checkpointing → Aggregation → Selective Unlearning**

This will provide a closer implementation of the original SISA methodology.

---

# Recommended Next Experiments

1. Complete the pure-SISA model comparison.
2. Standardize the experiment configuration across all classifiers.
3. Correct script labels so every output identifies the actual algorithm being tested.
4. Use the same training/test split and random seed for every comparison.
5. Measure full training time and unlearning time using the same timing methodology.
6. Run multiple forgotten samples rather than relying on one sample.
7. Report mean and standard deviation for accuracy, F1, and unlearning time.
8. Compare SISA unlearning against equivalent full retraining.
9. Implement slicing and checkpointing.
10. Measure the actual amount of retraining work avoided by checkpoint-based SISA.
11. Compare the SISA + Fine-Tuning hybrid against the checkpoint-based SISA implementation.
12. Evaluate whether the hybrid improves unlearning efficiency without sacrificing predictive performance.
13. Test the hybrid using multiple classifiers and multiple forget sets.
14. Add the pretrained-model + SISA approach to the final experimental comparison after implementing its unlearning stage.

---

# Final Research Direction

The project will evaluate the trade-off between:

**Predictive Accuracy ↔ Training Cost ↔ Unlearning Cost ↔ Model Architecture**

The final progression is:

```text
Conventional Model
        ↓
Pure SISA
        ↓
Multiple Learning Algorithms
        ↓
Sample-Level Unlearning
        ↓
Sliced + Checkpoint SISA
        ↓
SISA + Fine-Tuning Hybrid
        ↓
Additional Unlearning Methods
        ↓
Heterogeneous/Hybrid SISA
        ↓
Final Utility–Efficiency Comparison
```

The central objective is to determine whether SISA-based approaches can reduce the computational cost of machine unlearning while maintaining competitive predictive performance, and whether combining SISA with fine-tuning can provide a useful hybrid approach for selective model updating.

### Current primary proposed method

> **SISA + Fine-Tuning Hybrid: Shared Base Model → SISA Sharding → Shard-Specific Fine-Tuning → Selective Sample Unlearning → Prediction Aggregation**

This is the **main hybrid architecture developed in the project**, while the current experimental results should be reported as comparative evidence rather than as proof that the hybrid is universally superior.
