# Student Dropout Prediction in Ghanaian Basic Schools Using Machine Learning

**KNUST – Department of Computer Science**  
**2025–2026 Academic Year**  
**Group 5**

A comprehensive machine learning framework for the early prediction of student dropout in Ghanaian basic schools using institutional records, baseline machine learning models, class imbalance handling (SMOTE and CTGAN), hyperparameter optimization, and explainable artificial intelligence (SHAP).

---

# Project Overview

Student dropout remains a significant challenge affecting educational outcomes in Ghana. Early identification of students at risk enables timely interventions by teachers, school administrators, and policymakers.

This project develops and evaluates multiple machine learning models to predict student dropout using demographic, academic, attendance, behavioural, and socioeconomic data collected from Ghanaian basic schools.

The study follows a reproducible end-to-end machine learning pipeline, beginning with data cleaning and exploratory analysis through model development, optimization, explainability, and comparative evaluation.

---

# Research Objectives

The project aims to:

- Develop machine learning models for early dropout prediction.
- Compare the performance of multiple classification algorithms.
- Investigate the effect of class imbalance handling using SMOTE and CTGAN.
- Optimize the best-performing model through hyperparameter tuning.
- Explain model predictions using SHAP.
- Produce a reproducible machine learning workflow suitable for educational research.

---

# Machine Learning Pipeline

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Baseline Models
      │
      ▼
SMOTE vs CTGAN Experiments
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Final Model
      │
      ▼
SHAP Explainability
      │
      ▼
Model Comparison & Evaluation
```

---

# Machine Learning Models

The following supervised learning algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The best-performing model is selected based on objective evaluation metrics before optimization.

---

# Class Imbalance Experiments

Two strategies are investigated:

### Experiment A

Original Dataset + SMOTE

### Experiment B

Original Dataset + CTGAN Synthetic Data

The effectiveness of both approaches is compared using identical evaluation procedures.

---

# Explainable AI

The final selected model is interpreted using SHAP.

Generated explanations include:

- Global feature importance
- SHAP summary plots
- Waterfall plots
- Force plots
- Individual prediction explanations

---

# Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- Cross Validation

---

# Repository Structure

```
student-dropout-prediction-ghana/
│
├── notebooks/
│   ├── Notebook_01_Data_Cleaning.ipynb
│   ├── Notebook_02_EDA.ipynb
│   ├── Notebook_03_Feature_Engineering.ipynb
│   ├── Notebook_04_Baseline_Models.ipynb
│   ├── Notebook_05_Class_Imbalance_Experiments.ipynb
│   ├── Notebook_06_Model_Optimization.ipynb
│   ├── Notebook_07_Model_Explainability.ipynb
│   └── Notebook_08_Final_Evaluation.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── figures/
│
├── models/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- CTGAN
- SDV
- SHAP
- Imbalanced-Learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab

---

# Reproducibility

The project follows best practices for reproducible machine learning:

- Fixed random seeds
- Stratified train-test split
- No data leakage
- SMOTE applied only to training data
- Independent test set
- Pipeline-based preprocessing
- Saved trained models
- Version-controlled notebooks

---

# Team

| Name | Student ID | Role |
|------|------------|------|
| Oheneba Kwaku Tawiah Ntim | 20923785 | Lead Researcher & Machine Learning Development |
| Jude Ahiekpor Kekeli Yao | 20920037 | Literature Review & Baseline Models |
| Evangelina Temple | 20917568 | Data Collection & Preprocessing |
| Victoria Teye | 20920301 | Model Training & Evaluation |
| Kyei Christian Junior | 20923927 | Explainable AI & Results Analysis |

**Supervisor:**  
Dr. Eric Opoku Osei  
Department of Computer Science  
Kwame Nkrumah University of Science and Technology (KNUST)

---

# Ethics

The study uses anonymized educational records collected from participating Ghanaian basic schools.

All analyses comply with institutional ethical requirements and applicable data protection regulations.

Confidential student information is excluded from this public repository.

---

# Citation

If you use this repository in academic work, please cite:

> Oheneba Kwaku Tawiah Ntim et al. (2026). *Student Dropout Prediction in Ghanaian Basic Schools Using Machine Learning*. Kwame Nkrumah University of Science and Technology.

---

# License

This project is released under the KNUST License.

---

# Status

**Current Phase**

- Data Cleaning ✓
- Exploratory Data Analysis ✓
- Feature Engineering ✓
- Baseline Models ✓
- Class Imbalance Experiments ✓
- Hyperparameter Optimization ✓
- Explainable AI (SHAP) ✓
- Final Evaluation ✓
