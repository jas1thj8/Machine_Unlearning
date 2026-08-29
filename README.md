# Machine Unlearning: SISA and Fine-Tuning Hybrids

This project evaluates machine-unlearning methods on the Wisconsin breast-cancer dataset. It compares standard **SISA** (Sharded, Isolated, Sliced, and Aggregated) classifiers with **SISA + fine-tuning hybrid** classifiers.

## Methods

### Original SISA

The training data is divided into three stratified shards. One independent classifier is trained for each shard, and class probabilities from all shard models are averaged for the final prediction. When a training sample must be forgotten, only the model for the affected shard is retrained without that sample.

The original SISA implementations are:

- Logistic Regression: `sisaLR.py`
- Random Forest: `sisaRF.py`
- Support Vector Machine: `sisaSVM.py`
- Decision Tree: `sisaDT.py`
- K-Nearest Neighbors: `sisaKNN.py`
- Gaussian Naive Bayes: `sisaNB.py`

### SISA + Fine-Tuning Hybrid

The hybrid first trains a shared base model on 20% of the training split. A copy of that base model is then adapted using each of three SISA shards. Predictions are aggregated by averaging probabilities. For unlearning, only the affected child model is rebuilt from the unchanged base model and its updated shard.

Run every hybrid model with:

```powershell
python sisa_finetuning_hybrid_classical.py --model all
```

Run a single hybrid, for example:

```powershell
python sisa_finetuning_hybrid_classical.py --model decision_tree
```

## Accuracy Comparison

The results below are the initial held-out test accuracy from the current experiments.

| Classifier | Original SISA | SISA + fine-tuning hybrid | Difference |
|---|---:|---:|---:|
| Logistic Regression | 98.25% | 98.25% | 0.00 pp |
| Random Forest | 95.61% | 94.74% | -0.88 pp |
| SVM | 98.25% | 96.49% | -1.75 pp |
| Decision Tree | 93.86% | 96.49% | +2.63 pp |
| KNN | 94.74% | 93.86% | -0.88 pp |
| Gaussian Naive Bayes | 92.11% | 92.11% | 0.00 pp |

## Findings

- The highest observed original-SISA accuracy is **98.25%**, achieved by Logistic Regression and SVM.
- The highest hybrid accuracy is **98.25%**, achieved by Logistic Regression.
- Fine-tuning improves the Decision Tree result by **2.63 percentage points**.
- Fine-tuning preserves Logistic Regression and Gaussian Naive Bayes accuracy, but reduces accuracy slightly for Random Forest, SVM, and KNN in this experiment.

## Comparison Note

The comparison uses the same dataset, random seed, stratified train/test split, and number of SISA shards. It is not a perfectly identical training-budget comparison: original SISA trains its shard models using the whole training split, while the hybrid reserves 20% for common base-model pretraining and adapts the remaining data in shards. This design is intentional because the shared base model is the hybrid's fine-tuning starting point.
