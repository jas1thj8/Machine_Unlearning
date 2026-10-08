# Machine Unlearning using Classical Machine Learning

A research project focused on **Machine Unlearning using Classical Machine Learning models**. The project investigates how selected training data can be removed from trained models while maintaining model performance and reducing the computational cost compared with complete model retraining.

## Project Evolution

### Initial Dataset — Breast Cancer Wisconsin

The project initially used the **Breast Cancer Wisconsin dataset** to develop and experiment with different machine unlearning approaches.

The initial work explored **SISA, Fine-Tuning, Gradient-Based Unlearning, Influence-Based Unlearning, and Hybrid SISA + Fine-Tuning approaches**.

These experiments helped establish the initial machine unlearning pipeline and identify limitations in the dataset and experimental setup.

The previous implementations and experiments are preserved in the **old_dataset** directory.

---

# Current Dataset — UCI Human Activity Recognition

The current research uses the **UCI Human Activity Recognition Using Smartphones (UCI-HAR)** dataset.

The dataset contains smartphone sensor measurements collected from subjects performing different physical activities.

### Dataset Characteristics

| Property | Value |
|---|---:|
| Total samples | 10,299 |
| Training samples | 7,352 |
| Testing samples | 2,947 |
| Features | 561 |
| Classes | 6 |
| Training subjects | 21 |
| Testing subjects | 9 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Activity Classes

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

The original subject-based training and testing separation provided by the UCI dataset is preserved.

---

# Data Preprocessing

The original UCI-HAR dataset was processed into structured Parquet datasets for the machine learning experiments.

The preprocessing included loading the feature definitions, training and testing feature matrices, activity labels, and subject identifiers.

The processed datasets were validated for missing values, duplicate records, class labels, and subject separation.

The subject information is retained for machine unlearning experiments but is not used as a predictive feature.

---

# Classical Machine Learning Baselines

Before implementing machine unlearning, eight classical machine learning algorithms were evaluated on the UCI-HAR dataset.

| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|---|---:|---:|---:|---:|---:|
| **Linear SVM** | **96.71%** | **96.97%** | **96.68%** | **96.74%** | **1.13 s** |
| **Logistic Regression** | **96.06%** | **96.31%** | **95.98%** | **96.06%** | **2.55 s** |
| Gradient Boosting | 93.89% | 94.07% | 93.73% | 93.83% | 502.59 s |
| XGBoost | 92.74% | 92.91% | 92.55% | 92.65% | 13.90 s |
| Random Forest | 92.57% | 92.71% | 92.28% | 92.41% | 1.09 s |
| KNN | 90.16% | 90.67% | 89.68% | 89.85% | 0.06 s |
| Decision Tree | 86.22% | 86.25% | 85.87% | 85.95% | 2.90 s |
| Naive Bayes | 77.03% | 79.25% | 76.91% | 76.72% | 0.09 s |

Linear SVM and Logistic Regression produced the strongest baseline results while remaining computationally practical. Logistic Regression was selected for the initial SISA implementation.

---

# SISA Machine Unlearning

The current machine unlearning implementation focuses on **SISA — Sharded, Isolated, Sliced, and Aggregated learning**.

The UCI-HAR training data was divided into **three subject-level shards**.

Subject-level sharding ensures that all samples belonging to the same subject remain within a single shard. This is important for subject-level machine unlearning.

### Shard Distribution

| Shard | Subjects | Samples |
|---|---|---:|
| Shard 1 | 1, 6, 11, 16, 21, 25, 28 | 2,553 |
| Shard 2 | 3, 7, 14, 17, 22, 26, 29 | 2,397 |
| Shard 3 | 5, 8, 15, 19, 23, 27, 30 | 2,402 |
| **Total** | **21 subjects** | **7,352** |

For the current experiment, **Subject 1** was selected as the forget set.

---

# Original SISA Training

Each shard was independently trained using Logistic Regression.

| Shard | Samples | Training Accuracy | Training Time |
|---|---:|---:|---:|
| Shard 1 | 2,553 | 99.49% | 0.37 s |
| Shard 2 | 2,397 | 99.92% | 0.25 s |
| Shard 3 | 2,402 | 99.92% | 0.22 s |
| **Total** | **7,352** | — | **0.84 s** |

The individual shard models were aggregated using averaged class probabilities.

### Original SISA Test Performance

| Metric | Result |
|---|---:|
| Accuracy | **94.91%** |
| Precision | **95.13%** |
| Recall | **94.91%** |
| F1 Score | **94.91%** |

The original Logistic Regression model achieved 96.06% accuracy, while SISA achieved 94.91%, resulting in a difference of approximately 1.15 percentage points.

---

# Unlearning Subject 1

Subject 1 was selected as the forget set.

The forget and retain datasets contain:

| Dataset | Samples |
|---|---:|
| Forget Set — Subject 1 | 347 |
| Retain Set | 7,005 |

Subject 1 belongs to Shard 1.

The original Shard 1 contained 2,553 samples. After removing Subject 1, 2,206 samples remained.

Only the affected shard was retrained. Shards 2 and 3 were left unchanged.

### SISA Unlearning Time

**Approximately 0.43 seconds**

---

# SISA After Unlearning

After Subject 1 was removed and Shard 1 was retrained, the three shard models were aggregated again.

| Metric | Original SISA | SISA After Unlearning |
|---|---:|---:|
| Accuracy | 94.91% | **94.64%** |
| Precision | 95.13% | **94.88%** |
| Recall | 94.91% | **94.64%** |
| F1 Score | 94.91% | **94.62%** |

The test accuracy changed from **94.91% to 94.64%**, a decrease of approximately **0.27 percentage points**.

This indicates that removing Subject 1 produced only a small change in overall test performance.

---

# Full Retraining Reference

To compare SISA unlearning against conventional machine retraining, a Logistic Regression model was completely retrained using the 7,005 retained samples.

### Full Retraining Results

| Metric | Result |
|---|---:|
| Accuracy | **96.13%** |
| Precision | **96.27%** |
| Recall | **96.13%** |
| F1 Score | **96.12%** |
| Retraining Time | **2.01 s** |

---

# SISA Unlearning vs Full Retraining

| Metric | SISA Unlearning | Full Retraining |
|---|---:|---:|
| Samples retrained | 2,206 | 7,005 |
| Time | **0.43 s** | **2.01 s** |
| Test Accuracy | 94.64% | **96.13%** |
| Test F1 Score | 94.62% | **96.12%** |

The approximate speedup of SISA unlearning over full retraining is:

**2.01 / 0.43 ≈ 4.64×**

Therefore, the current experiment shows that SISA can perform the required model update substantially faster than retraining the complete model.

---

# Forgetting Evaluation

To evaluate the effect of removing Subject 1, both the forget set and retain set were evaluated.

The evaluation compared:

- Original SISA
- SISA after unlearning
- Full retraining

## Forget Set Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Original SISA | 100.00% | 100.00% | 100.00% | 100.00% |
| SISA After Unlearning | 99.42% | 99.52% | 99.32% | 99.41% |
| Full Retraining | 99.71% | 99.69% | 99.66% | 99.67% |

## Retain Set Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Original SISA | 96.05% | 96.13% | 96.08% | 96.07% |
| SISA After Unlearning | 95.97% | 96.05% | 95.98% | 95.98% |
| Full Retraining | 99.20% | 99.27% | 99.27% | 99.27% |

An important observation from this experiment is that **forget-set accuracy alone cannot be used as sufficient evidence of successful machine unlearning**, because a model trained without the forgotten subject can still classify those samples accurately.

For this reason, the unlearned SISA model was additionally compared with a SISA model retrained without Subject 1.

---

# SISA Retraining Without Subject 1

A separate reference experiment completely retrained all three SISA shards after removing Subject 1.

| Shard | Samples | Training Time |
|---|---:|---:|
| Shard 1 | 2,206 | 0.4519 s |
| Shard 2 | 2,397 | 0.4074 s |
| Shard 3 | 2,402 | 0.3976 s |
| **Total** | **7,005** | **1.2569 s** |

### Forget Set

- Accuracy: **99.42%**
- Precision: **99.52%**
- Recall: **99.32%**
- F1 Score: **99.41%**

### Retain Set

- Accuracy: **95.97%**
- Precision: **96.05%**
- Recall: **95.98%**
- F1 Score: **95.98%**

The results exactly matched the SISA-after-unlearning results under the implemented evaluation.

---

# Model Similarity Evaluation

The SISA model after unlearning was compared with the SISA model completely retrained without Subject 1.

The comparison included prediction agreement, mean probability difference, and model coefficient difference.

| Comparison | Result |
|---|---:|
| Forget Set Prediction Agreement | **1.000000** |
| Retain Set Prediction Agreement | **1.000000** |
| Test Set Prediction Agreement | **1.000000** |
| Forget Set Probability Difference | **0.00000000** |
| Retain Set Probability Difference | **0.00000000** |
| Test Set Probability Difference | **0.00000000** |
| Shard 1 Coefficient Difference | **0.00000000** |
| Shard 2 Coefficient Difference | **0.00000000** |
| Shard 3 Coefficient Difference | **0.00000000** |

The SISA model after unlearning **exactly matched the SISA model retrained without Subject 1 under the implemented prediction, probability, and coefficient comparisons**.

---

# Project Structure

The current project is organized into two main sections.

### New Dataset

The **new_dataset** directory contains the current UCI-HAR research work.

It includes the original UCI-HAR dataset, processed datasets, trained models, baseline experiments, SISA implementation, unlearning experiments, forgetting evaluation, retraining experiments, and model similarity analysis.

### Old Dataset

The **old_dataset** directory contains the previous Breast Cancer based experiments and earlier machine unlearning implementations.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Parquet
- Git
- GitHub

---

# Current Project Status

### Completed

- UCI-HAR dataset selection
- Dataset preprocessing
- Dataset validation
- Eight classical ML baseline experiments
- Logistic Regression SISA implementation
- Subject-level SISA sharding
- Subject-level forgetting
- SISA unlearning
- Full retraining comparison
- Forget-set evaluation
- Retain-set evaluation
- SISA retraining without forgotten data
- Model similarity evaluation

### Next Stage

The next stage of the project will extend the current framework to additional machine unlearning approaches and compare them with SISA and full retraining.

The planned evaluation will focus on:

- Model accuracy
- Precision
- Recall
- F1 Score
- Forgetting effectiveness
- Retain-set performance
- Unlearning time
- Full retraining time
- Model similarity

---

# Author

**Jaswanth Talla**

B.Tech Computer Science and Engineering — AI & Data Science

**Project:** Machine Unlearning using Classical Machine Learning
