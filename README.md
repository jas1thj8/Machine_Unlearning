# Machine Unlearning: SISA and Fine-Tuning Hybrids

This project evaluates machine-unlearning methods on the Wisconsin Breast Cancer Diagnostic dataset. It compares standard **SISA (Sharded, Isolated, Sliced, and Aggregated)** classifiers with **SISA + fine-tuning hybrid** classifiers, while investigating how different learning algorithms affect predictive performance and unlearning cost.

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

### SISA + Fine-Tuning Hybrid

The hybrid approach first trains a shared base model on 20% of the training split. A copy of this base model is then adapted using each SISA shard.

The predictions from the adapted shard models are aggregated to obtain the final prediction.

For unlearning, the affected child model is rebuilt from the unchanged base model using the updated shard, while the other child models remain unchanged.

Run all hybrid models with:

```powershell
python sisa_finetuning_hybrid_classical.py --model all
```

Run a single hybrid model, for example:

```powershell
python sisa_finetuning_hybrid_classical.py --model decision_tree
```

---

## Original SISA Results

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

The Logistic Regression experiment achieved 98.25% accuracy and 98.24% F1.

SVM also achieved 98.25% accuracy and 98.24% F1.

XGBoost achieved 98.25% accuracy and 98.24% F1.

---

## SISA Unlearning Results

The current experiments perform sample-level unlearning by identifying the affected shard and retraining that shard.

| Classifier           | Before Unlearning | After Unlearning |       Change | Unlearning Time |
| -------------------- | ----------------: | ---------------: | -----------: | --------------: |
| Logistic Regression  |            98.25% |           98.25% |      0.00 pp |          0.01 s |
| Random Forest        |            95.61% |           95.61% |      0.00 pp |          0.21 s |
| SVM                  |            98.25% |           97.37% | **-0.88 pp** |         ~0.00 s |
| Decision Tree        |            93.86% |           93.86% |      0.00 pp |         ~0.00 s |
| KNN                  |            94.74% |           94.74% |      0.00 pp |         ~0.00 s |
| Gaussian Naive Bayes |            92.11% |       **92.98%** | **+0.87 pp** |         ~0.00 s |
| Gradient Boosting    |            95.61% |           95.61% |      0.00 pp |          0.13 s |
| XGBoost              |            98.25% |           98.25% |      0.00 pp |          0.05 s |

The SVM experiment is the only current run showing a decrease in accuracy after the selected sample was forgotten.

Gaussian Naive Bayes increased from 92.11% to 92.98% after unlearning.

For Gradient Boosting, accuracy remained 95.61% after unlearning, with an unlearning time of 0.13 seconds.

For XGBoost, accuracy remained 98.25%, with an unlearning time of 0.05 seconds.

---

## Accuracy Comparison: Original SISA vs Fine-Tuning Hybrid

The previously obtained hybrid results are:

| Classifier           | Original SISA | SISA + Fine-Tuning Hybrid |   Difference |
| -------------------- | ------------: | ------------------------: | -----------: |
| Logistic Regression  |        98.25% |                    98.25% |      0.00 pp |
| Random Forest        |        95.61% |                    94.74% |     -0.88 pp |
| SVM                  |        98.25% |                    96.49% |     -1.75 pp |
| Decision Tree        |        93.86% |                **96.49%** | **+2.63 pp** |
| KNN                  |        94.74% |                    93.86% |     -0.88 pp |
| Gaussian Naive Bayes |        92.11% |                    92.11% |      0.00 pp |

The hybrid therefore does not provide a universal accuracy improvement. Its largest observed improvement is for Decision Tree, while Logistic Regression and Gaussian Naive Bayes remain unchanged.

---

## Current Findings

* The highest observed pure-SISA accuracy is **98.25%**, achieved by **Logistic Regression, SVM and XGBoost**.
* The highest reported hybrid accuracy is also **98.25%**, achieved by **Logistic Regression**.
* Fine-tuning improves the Decision Tree result by **2.63 percentage points**.
* Fine-tuning preserves Logistic Regression and Gaussian Naive Bayes accuracy in the current comparison.
* Fine-tuning decreases the reported accuracy of Random Forest, SVM and KNN.
* Most current unlearning experiments produce **no change in held-out accuracy**.
* SVM shows a measurable decrease after unlearning.
* Gaussian Naive Bayes shows a small increase after unlearning.
* The current experiments demonstrate that the effect of forgetting a sample depends on both the selected sample and the learning algorithm.

---

## Important Methodological Update

The current implementation should be distinguished from the **full checkpoint-based SISA architecture**.

The present experiments perform **shard-level retraining**: after identifying the affected shard, that shard is retrained without the forgotten sample.

Full SISA additionally uses **slicing and checkpointing**. A shard is divided into ordered slices, intermediate model states are saved, and unlearning can restart from an appropriate checkpoint rather than retraining the entire affected shard.

Therefore, the current experiments should be described as:

> **Pure SISA baseline with shard-level sample unlearning**

The next implementation stage is:

> **Sliced and checkpoint-based SISA for selective retraining**

---

## Current Research Structure

The project is now organized into four major layers:

### Layer 1 — Conventional Baseline

Train a conventional full-data model to establish reference predictive performance and training cost.

### Layer 2 — Pure SISA

Train the same learning algorithm independently across the SISA shards and compare different algorithms.

Current models:

**Logistic Regression → Random Forest → SVM → Decision Tree → KNN → Gaussian NB → Gradient Boosting → XGBoost**

### Layer 3 — SISA + Fine-Tuning Hybrid

Use a shared base model and adapt copies of it to individual SISA shards. Compare the hybrid against the corresponding pure-SISA models.

### Layer 4 — Full SISA

Introduce:

**Sharding → Isolation → Slicing → Checkpointing → Aggregation → Selective Unlearning**

This will provide a closer implementation of the original SISA methodology and allow a more meaningful analysis of unlearning efficiency.

---

## Recommended Next Experiments

1. **Complete the pure-SISA model comparison.**
2. Standardize the experiment configuration across all classifiers.
3. Correct the script labels so every output identifies the actual algorithm being tested.
4. Measure full training time and unlearning time using the same timing methodology.
5. Run multiple forgotten samples rather than relying on a single sample.
6. Measure the average and standard deviation of accuracy changes after unlearning.
7. Compare SISA unlearning against equivalent full-model retraining.
8. Implement slicing and checkpointing.
9. Measure the actual reduction in retraining work provided by checkpoint-based SISA.
10. After validating pure SISA, continue with the fine-tuning/heterogeneous SISA experiments.

---

## Final Research Direction

The project will ultimately evaluate the trade-off between:

**Predictive Accuracy ↔ Training Cost ↔ Unlearning Cost ↔ Model Architecture**

The intended progression is:

**Conventional Model**
↓
**Pure SISA**
↓
**Multiple Learning Algorithms**
↓
**Sample-Level Unlearning**
↓
**Sliced + Checkpoint SISA**
↓
**SISA + Fine-Tuning Hybrid**
↓
**Heterogeneous/Hybrid SISA**
↓
**Final Utility–Efficiency Comparison**

The central objective is to determine whether SISA can substantially reduce the cost of machine unlearning while maintaining competitive predictive performance, and whether the choice or combination of learning algorithms can further improve this trade-off.
