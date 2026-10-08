You're looking for a **proper project README**, not a report-style summary. It should read like a GitHub repository: concise, technical, and focused mainly on the **current UCI-HAR work**, while treating the previous dataset only as project history.

Use this:

```markdown
# Machine Unlearning

A research-oriented implementation and experimentation project focused on **Machine Unlearning using Classical Machine Learning models**.

The project investigates how selected training data can be removed from a trained model while preserving the model's overall performance. Different unlearning strategies are being explored and evaluated against conventional full retraining.

---

## Project Evolution

### Previous Work — Breast Cancer Dataset

The initial stage of the project used the **Breast Cancer Wisconsin dataset** to experiment with classical machine learning and different machine unlearning approaches.

The experiments included:

- SISA
- Fine-Tuning
- Gradient-Based Unlearning
- Influence-Based Unlearning
- Hybrid Unlearning approaches

The initial experiments helped establish the project structure and identify practical limitations of using smaller datasets for evaluating different unlearning techniques.

The previous implementations and experiments are preserved in:

```text
old_dataset/
```

---

# Current Work — UCI Human Activity Recognition

The project has now moved to the **UCI Human Activity Recognition Using Smartphones (UCI-HAR)** dataset.

The UCI-HAR dataset contains smartphone sensor measurements collected from multiple subjects performing different physical activities.

### Dataset

- **10,299 total samples**
- **561 features**
- **6 activity classes**
- **21 training subjects**
- **9 testing subjects**
- No missing values
- No duplicate rows

### Activities

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

The original UCI-HAR training and testing subject split is preserved in the project.

---

# Classical Machine Learning Baselines

Before implementing unlearning, multiple classical machine learning algorithms were evaluated on the UCI-HAR dataset.

| Model | Accuracy |
|---|---:|
| Linear SVM | **96.71%** |
| Logistic Regression | **96.06%** |
| Gradient Boosting | **93.89%** |
| XGBoost | **92.74%** |
| Random Forest | **92.57%** |
| KNN | **90.16%** |
| Decision Tree | **86.22%** |
| Naive Bayes | **77.03%** |

These baseline experiments provide a reference for evaluating the effect of machine unlearning on model performance.

---

# SISA Machine Unlearning

The current implementation focuses on **SISA (Sharded, Isolated, Sliced, and Aggregated)** machine unlearning.

The UCI-HAR training data is divided into **3 subject-level shards**.

Subject-level sharding is used so that all samples belonging to a particular subject remain within the same shard.

```text
Training Dataset
       │
       ├── Shard 1
       ├── Shard 2
       └── Shard 3
```

For the current experiment, **Subject 1** is selected as the forget set.

When Subject 1 needs to be removed:

1. Identify the shard containing Subject 1.
2. Remove Subject 1 from that shard.
3. Retrain only the affected shard.
4. Keep the other shards unchanged.
5. Aggregate the predictions from all shards.

This avoids retraining the complete model from scratch.

---

# Current SISA Results

### Original SISA

- Accuracy: **94.91%**
- F1 Score: **94.91%**

### SISA After Unlearning Subject 1

- Accuracy: **94.64%**
- F1 Score: **94.62%**
- Unlearning time: **~0.43 seconds**

### Full Retraining

- Accuracy: **96.13%**
- F1 Score: **96.12%**
- Retraining time: **~2.01 seconds**

SISA therefore provides a substantially smaller retraining workload for the current experiment.

The unlearned SISA model was additionally compared with a SISA model retrained from scratch after removing Subject 1. The implemented comparison produced:

- **100% prediction agreement**
- **0 mean probability difference**
- **0 mean coefficient difference**

---

# Project Structure

```text
MUL/
│
├── new_dataset/
│   │
│   ├── UCI HAR Dataset/
│   │   └── Original UCI-HAR dataset
│   │
│   ├── data/
│   │   ├── uci_har_train.parquet
│   │   ├── uci_har_test.parquet
│   │   ├── sisa_shard_1.parquet
│   │   ├── sisa_shard_2.parquet
│   │   ├── sisa_shard_3.parquet
│   │   ├── sisa_forget_subject_1.parquet
│   │   └── sisa_retain_subject_1.parquet
│   │
│   ├── models/
│   │   ├── sisa_shard_1_unlearned.joblib
│   │   └── full_retrain_subject_1.joblib
│   │
│   └── scripts/
│       └── uci_hair_dataset/
│           ├── Baseline ML experiments
│           ├── SISA sharding
│           ├── SISA training
│           ├── SISA unlearning
│           ├── Full retraining
│           ├── Forgetting evaluation
│           └── Model similarity evaluation
│
└── old_dataset/
    └── Previous Breast Cancer experiments
```

---

# Objectives

The main objectives of this project are:

- Implement machine unlearning using classical ML models.
- Study the effectiveness of SISA-based unlearning.
- Compare unlearning with full model retraining.
- Measure the computational cost of unlearning.
- Evaluate model performance after removing selected training data.
- Measure forgetting effectiveness using forget and retain sets.
- Compare unlearned models with models retrained without the forgotten data.
- Extend the framework to additional machine unlearning techniques.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Parquet
- Git / GitHub

---

# Current Status

**Dataset:** UCI-HAR  
**Learning approach:** Classical Machine Learning  
**Unlearning method:** SISA  
**SISA shards:** 3  
**Forget subject:** Subject 1  
**Baseline experiments:** Completed  
**SISA implementation:** Completed  
**Full retraining comparison:** Completed  
**Forgetting evaluation:** Completed  
**Model similarity evaluation:** Completed  

### Next Stage

The next stage of the project will extend the current framework to additional machine unlearning methods and compare them using common evaluation criteria such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Forgetting effectiveness
- Retain-set performance
- Unlearning time
- Full retraining time
- Model similarity

---

## Author

**Jaswanth Talla**

B.Tech — Computer Science & Engineering (AI & Data Science)
```
