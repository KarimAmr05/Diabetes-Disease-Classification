# Diabetes Disease Classification

End‑to‑end machine learning pipeline for diabetes prediction using patient health data. Includes domain‑specific outlier filtering, SMOTE for class imbalance, per‑model threshold tuning, and comprehensive evaluation (F1‑score, ROC‑AUC, PR‑AUC). Compares Logistic Regression, Gaussian Naive Bayes, MLP, SVM (RBF/Linear), Random Forest, and XGBoost. The final model (XGBoost) achieves **0.8090 test F1‑score**.

<p align="center">
	<a href="Diabetes_Classification.ipynb"><img src="https://img.shields.io/badge/Open%20Notebook-Diabetes_Classification.ipynb-0f766e?style=for-the-badge&logo=jupyter" alt="Open Notebook" /></a>
	<a href="#project-overview"><img src="https://img.shields.io/badge/Task%202%20ML-Classification-20BEFF?style=for-the-badge" alt="Task 2 ML" /></a>
	<a href="#getting-started"><img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
	<a href="#key-highlights"><img src="https://img.shields.io/badge/SMOTE-Imbalance%20Handling-2e7d32?style=for-the-badge" alt="SMOTE" /></a>
	<a href="#key-highlights"><img src="https://img.shields.io/badge/Threshold%20Tuning-F1%20Optimization-8B5CF6?style=for-the-badge" alt="Threshold Tuning" /></a>
</p>

## Project Overview

This repository contains a complete workflow for diabetes disease classification: data cleaning (duplicate removal, clinical range filtering), feature engineering, SMOTE‑based balancing, per‑model threshold tuning, hyperparameter optimization (GridSearchCV, Optuna), and final evaluation. Because the dataset is imbalanced (91.5% non‑diabetic, 8.5% diabetic), **F1‑score** is used as the primary metric. The best performing model is **XGBoost** with a test F1‑score of **0.8090**.

## Key Highlights

- **Duplicate removal**: 3,854 duplicate rows eliminated → 96,134 clean records.
- **Clinical outlier filtering**: Age (0–110), BMI (10–75), HbA1c (3–16), glucose (30–550) — medically meaningful bounds.
- **Stratified split**: 70% train / 15% validation / 15% test (stratified by diabetes class).
- **SMOTE** applied only to training set (sampling_strategy=0.7) to avoid data leakage.
- **Threshold tuning** for every model (search 0.001–0.999) to maximise validation F1‑score.
- **PCA analysis**: 5 components retained 89.4% variance but degraded XGBoost F1 from 0.8090 → 0.7517 → not used in final model.
- **Clinical feature engineering** (HbA1c≥6.5, glucose≥200, BMI≥30, age≥45, interactions) improved Logistic Regression and GNB.

## Workflow

1. Load and inspect dataset (`diabetes_prediction_dataset.csv`).
2. Clean data: remove duplicates, apply clinical range filters.
3. Train/validation/test split (stratified).
4. Preprocessing: standard scaling, one‑hot encoding, optional clinical features.
5. Apply SMOTE only to training data.
6. Train and hyperparameter‑tune models using validation set.
7. Optimise decision threshold per model on validation set.
8. Final evaluation on unseen test set (F1, precision, recall, ROC‑AUC, PR‑AUC, confusion matrices).
9. Learning curves and PCA impact analysis.

## Models Evaluated

- Logistic Regression (with clinical feature set)
- Gaussian Naive Bayes (with clinical feature set)
- Support Vector Machine (RBF and Linear)
- Multi‑Layer Perceptron (MLP)
- Random Forest
- **XGBoost** (best performer)

## Results Summary

The final test set results (after threshold tuning) are summarised below. **XGBoost** achieved the highest F1‑score and was selected as the final model.

| Model | Test F1‑Score | Test Precision | Test Recall |
|-------|--------------:|---------------:|-------------:|
| **XGBoost** | **0.8090** | 0.82 | 0.80 |
| Random Forest | 0.7914 | 0.85 | 0.74 |
| MLP | 0.7900 | 0.79 | 0.79 |
| Logistic Regression | 0.7487 | 0.70 | 0.81 |
| Gaussian Naive Bayes | 0.6228 | – | – |

> **Note**: RBF SVM validation F1 was ~0.6603 (not computed on test due to runtime).

## Why F1‑Score Was Used

The dataset is highly imbalanced (91.5% non‑diabetic). Accuracy alone can be misleading (e.g., a “always predict non‑diabetic” model would achieve 91.5% accuracy but 0% recall for diabetics). F1‑score balances precision and recall, making it the most informative metric for this medical classification task.

## Repository Structure

```text
.
├── Diabetes_Classification.ipynb     # Main notebook (cleaning, training, evaluation)
├── README.md                         # This file
├── dataset/
│   └── diabetes_prediction_dataset.csv
└── Documents/
    ├── Task_2_Diabetes_Classification.docx
    └── IEEE_Diabetes_Classification.pdf   # IEEE style paper# Diabetes-Disease-Classification
Diabetes disease classification with preprocessing, PCA &amp; PSO feature reduction, and SMOTE-based imbalance handling, along with cross-validation, hyperparameter tuning, and learning-curve analysis. Compares Logistic Regression, Gaussian Naive Bayes, Random Forest, SVM, MLP, and XGBoost using cross-validated evaluation. (Task 2 ML)
