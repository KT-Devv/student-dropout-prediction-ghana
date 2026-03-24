# LightGBM-SHAP Ensemble for Early Dropout Detection in Ghanaian Basic Schools

**KNUST CANDO Lab · 2025–2026 · Group 5**

A CTGAN-augmented institutional data framework for early identification
of at-risk pupils in Ghanaian Primary and Junior High Schools, using a
LightGBM + Random Forest ensemble with SHAP TreeExplainer outputs
validated by Ghanaian teachers and GES district officers.

## Research Summary

| Item | Detail |
|---|---|
| Primary model | LightGBM 4.3 + Random Forest (scikit-learn 1.4) |
| XAI layer | SHAP TreeExplainer — global beeswarm + local waterfall plots |
| Dataset | Institutional records from ≥10 Ghanaian basic schools + CTGAN augmentation |
| Fairness | Equal Opportunity & Equalized Odds gaps by gender and urban/rural zone |
| Target journal | Computers & Education: AI (Elsevier) |
| Backup journal | Education & Information Technologies (Springer) |

## Repository Structure

| Folder | Contents |
|---|---|
| `/data-raw/` | Original school register records — **never committed, protected by .gitignore** |
| `/data-processed/` | Cleaned, encoded, SMOTE-balanced datasets ready for training |
| `/notebooks/` | All experiment notebooks (see below) |
| `/models/` | Saved LightGBM and RF model checkpoints (joblib format) |
| `/results/` | Metric outputs, SHAP plots, fairness tables, ablation results |
| `/paper/` | Draft manuscript files |

## Notebooks (to be added during experiment phase)

| Notebook | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Raw register → cleaned dataset pipeline |
| `02_ctgan_augmentation.ipynb` | CTGAN training, SDMetrics validation, TSTR test |
| `03_baselines.ipynb` | Logistic Regression, Decision Tree, XGBoost baselines |
| `04_lightgbm_training.ipynb` | LightGBM + RF ensemble, hyperparameter grid search |
| `05_shap_analysis.ipynb` | Global beeswarm plots, local waterfall plots, teacher validation prep |
| `06_fairness_evaluation.ipynb` | Equal Opportunity and Equalized Odds gaps by gender and zone |
| `07_ablation_study.ipynb` | Four ablation conditions |

## Setup
```bash
# Option 1 — pip
pip install -r requirements.txt

# Option 2 — conda
conda env create -f environment.yml
conda activate ghana-dropout-shap
```

## Reproducibility

All experiments use 5 random seeds: **42, 123, 456, 789, 1024**.
All metrics are reported as mean ± standard deviation across seeds.
Model checkpoints are saved after each cross-validation fold using `joblib.dump()`.

## Team

| Name | Student ID | Role |
|---|---|---|
| Oheneba Kwaku Tawiah Ntim | 20923785 | Lead researcher |
| Jude Ahiekpor Kekeli Yao | 20920037 | Literature & baselines |
| Evangelina Temple | 20917568 | Data collection & preprocessing |
| Victoria Teye | 20920301 | Model training & evaluation |
| Kyei Christian Junior | 20923927 | SHAP analysis & fairness |

**Supervisor:** Eric Opoku Osei (PhD), Department of Computer Science, KNUST, Ghana

## Ethics

This study was approved by the KNUST Committee on Human Research,
Publications and Ethics (CHRPE). Reference No: [insert upon receipt].
All pupil data anonymised. Ghana Data Protection Act (2012) compliant.

## Citation

[To be added upon publication]
```

Commit message: `Update README — Path B LightGBM-SHAP architecture`

---

**STEP 6 — Add .gitignore to protect pupil data**

**"Add file" → "Create new file"** → name it `.gitignore` → paste:
```
# CRITICAL — never commit raw pupil data
data-raw/
*.csv
*.xlsx
*.xls

# Python
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
.virtual_documents/

# Model files (large binaries)
*.pkl
*.joblib

# Environment
.env
venv/
.venv/
```

Commit message: `Add .gitignore — protect raw pupil data`

---

**After all 6 steps your repo will look like this:**
```
ghana-basic-dropout-shap/
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
├── data-raw/
├── data-processed/
├── notebooks/
├── models/
├── results/
└── paper/