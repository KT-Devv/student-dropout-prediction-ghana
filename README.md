# Predicting School Dropout in Ghanaian Basic Schools with Focal-Loss LightGBM

Group 5 · R02 · Supervisor: Dr. Eric Opoku Osei
Department of Computer Science, KNUST

---

## Reproduce

```bash
git clone https://github.com/KT-Devv/student-dropout-prediction-ghana.git
cd student-dropout-prediction-ghana
pip install -r requirements.txt
# place the raw workbook at data-raw/ghana_dropout_study_M.xlsx (not committed — see Ethics)
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/Notebook_1_Data_Cleaning.ipynb \
  notebooks/Notebook_2_EDA.ipynb \
  notebooks/Notebook_3_Feature_Engineering.ipynb \
  notebooks/Notebook_4_Baselines.ipynb \
  notebooks/Notebook_5_Class_Imbalance.ipynb \
  notebooks/Notebook_5b_Data_Augmentation.ipynb \
  notebooks/Notebook_6_Tuning_Disclosed.ipynb \
  notebooks/Notebook_6b_Focal_Loss_ELightGBM.ipynb \
  notebooks/Notebook_7_SHAP_Signed.ipynb \
  notebooks/Notebook_9_Negative_Results_Diagnostic.ipynb
# then commit, set the two freeze flags in config.py, and run once:
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/Notebook_8_Final_Evaluation.ipynb
```

In Colab, clone the repo and set the environment variable instead of mounting
Drive:

```python
!git clone https://github.com/KT-Devv/student-dropout-prediction-ghana.git
import os; os.environ["DROPOUT_REPO"] = "/content/student-dropout-prediction-ghana"
```

No notebook calls `drive.mount()`. No path is hard-coded. Paths resolve
against the repository root, or against `DROPOUT_REPO` if set.

---

## Run order and what each notebook does

| # | Notebook | Reads | Writes | Touches test set? |
|---|---|---|---|---|
| 1 | Data Cleaning | `data-raw/*.xlsx` | `data-processed/cleaned_data.csv`, column cascade, missingness, outlier audit, cluster audit | no |
| 2 | EDA | `cleaned_data.csv` | descriptives, single-feature separation check | **no — training pool only** |
| 3 | Feature Engineering | `cleaned_data.csv` | composite specification, fold-safety checks, leak magnitude | no |
| 4 | Baselines | `cleaned_data.csv` | Supplementary Table S1 (six classifiers) | no |
| 5 | Class Imbalance | `cleaned_data.csv` | strategy comparison, validation-fold integrity check | no |
| 5b | Data Augmentation | `cleaned_data.csv` | SMOTE vs CTGAN, synthetic-data provenance | no |
| 6 | Tuning (disclosed) | `cleaned_data.csv` | every search trial, M13 disclosure table, orphan-artefact resolution | no |
| 6b | E-LightGBM | `cleaned_data.csv` | arm grid, ten seeds, γ×α grid, instance-level analysis | no |
| 7 | SHAP (signed) | `cleaned_data.csv` | signed attribution, family-level table, artefact reconciliation | no |
| 9 | Negative-Result Diagnostic | `cleaned_data.csv` | ten-seed stability, power, leave-one-seed-out, leave-one-school-out, Q23/Q24 | no |
| 8 | **Final Evaluation** | `cleaned_data.csv` | test-set scores, fairness, caseload, clearance certificate | **YES — gated, once** |

Notebook 8 is the only notebook that scores the held-out partition, and it
refuses to run unless `SCORE_TEST` and `FREEZE_CONFIRMED` are both `True` in
`config.py` and a git commit hash is available.

---

## Shared modules

Everything the notebooks share lives in three importable modules at the
repository root. Notebooks import; they never redefine.

- **`config.py`** — every path, column name, protocol constant and drop rule.
  The only place to change a column name.
- **`losses.py`** — the focal-loss objective. `make_focal(gamma, alpha)` for
  sweeps; top-level `focal_loss_lgb` / `focal_loss_eval` at the reported
  parameterisation, defined as real functions so they are picklable.
- **`pipeline.py`** — `preprocess_inside_fold()`, the arm grid, metrics,
  paired comparison, bootstrap CI, caseload translation.

This structure is deliberate. The R01 submission defined the focal loss in
three places, committed two SHAP artefacts that disagreed by a factor of two
with no label saying which model produced which, and used two different
definitions of `socioeconomic_vulnerability_score` in two notebooks — which
is why its two seed-42 results disagreed. Copy-paste is the defect.

---

## There is no engineered dataset

`data-processed/engineered_data_FOR_EDA_ONLY.csv` exists for figures and
descriptive tables. **No modelling notebook reads it.**

Feature engineering — imputation, encoding, scaling, and all three
composites — happens inside `pipeline.preprocess_inside_fold()`, fitted on
the training fold and applied to the held-out fold. An earlier version of
this project fitted those transforms on the full dataset and wrote the result
to `engineered_data.csv`, which a downstream notebook then split; every
encoder, scaler and median had already seen the held-out rows.

That cannot be fixed by patching the feature-engineering notebook, because
the defect is the *boundary*: as long as a fitted artefact crosses it as a
CSV, the leak exists. Hence the function.

---

## Protocol

- One frozen outer split: stratified 80/20 at `random_state=42`.
- All cross-validation runs on the **training pool only** —
  5 folds × 5 repeats × 10 seeds.
- Primary metric AUC-PR, with the base rate and the absolute positive count
  printed beside every figure.
- Headline contrast: `E_all_ce_noW` vs `G_all_focal_noW` — identical feature
  matrix, identical weighting mechanism, differing only in the `objective`
  argument.
- Class reweighting uses `sample_weight` for both losses, because
  `is_unbalance=True` is honoured only by LightGBM's built-in objectives and
  is inert under a custom one.
- Paired Wilcoxon signed-rank, Cohen's d, and a bootstrap CI computed in
  committed code.

Every run writes to `results/<notebook>/<RUN_ID>/` with a `RUN_MANIFEST.json`
recording the git commit, the environment, and the protocol constants. Cite
the run ID in M14, M19, M21 and every table caption.

---

## Ethics and data availability

Ethical approval: HuSSREC/AP/543/VOL. 5, Committee on Humanities and Social
Sciences Research and Ethics, KNUST. Valid 30 June 2026 – 30 June 2027.

Pupil-level data cannot be shared publicly and is **not committed**. Place the
raw workbook at `data-raw/` locally. Access to the primary field data may be
requested from the corresponding author subject to institutional ethics
approval and a data use agreement.

## Inference boundary

Findings are scoped to pupils in Primary 4 through JHS Form 2 at **four**
basic schools in the Kumasi Metropolitan area, Ashanti Region, Ghana. The
four schools contribute unevenly and their dropout rates differ substantially;
see `results/notebook01_cleaning/*/school_cluster_audit.csv`. No claim of
generalisability beyond these four sites is made.
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
│   ├── Notebook 1 — Data Cleaning & Preprocessing.ipynb
│   ├── Notebook 2 – Exploratory Data Analysis (EDA).ipynb
│   ├── Notebook 3 – Feature Engineering.ipynb
│   ├── Notebook 4 - Baselines.ipynb
│   ├── Notebook 5 – Class Imbalance Experiments (SMOTE vs CTGAN).ipynb
│   ├── Notebook 5b – Data Augmentation Experiments (SMOTE vs CTGAN).ipynb
│   ├── Notebook 6 – Model Engineering & Proposed Model.ipynb
│   ├── Notebook 6b - Focal Loss Engineering (E-LightGBM).ipynb
│   ├── Notebook 7 – Explainable AI (SHAP Analysis).ipynb
│   ├── Notebook 8 – Final Evaluation, Comparison & Dissertation Outputs.ipynb
│   └── Notebook 9 - Negative Results Diagnostic.ipynb
│
├── data-raw
│   
├── data-processed 
│   
│
├── figures/
│
├── models/
│
├── results/
│
├── losses.py
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
