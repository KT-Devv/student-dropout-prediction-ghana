# Federated GraphSAGE-SHAP Ensemble for Dropout Prediction in Ghanaian Basic Schools

**KNUST CANDO Lab · 2025–2026**

A privacy-preserving federated GNN framework for early identification
of at-risk pupils in Ghanaian Primary and Junior High Schools,
with SHAP-validated explanations for teachers and GES district officers.

## Repository Structure

| Folder | Contents |
|---|---|
| `/data-raw/` | Original school register records (anonymised, not tracked in git) |
| `/data-processed/` | Cleaned, encoded, graph-ready datasets |
| `/notebooks/` | Training, evaluation, SHAP analysis, and CTGAN augmentation notebooks |
| `/models/` | Saved model checkpoints (joblib / PyTorch state dicts) |
| `/federated/` | Flower server and client scripts for federated training |
| `/results/` | Metric outputs, SHAP plots, fairness tables |
| `/paper/` | Draft manuscript files |

## Setup
```bash
# Option 1 — pip
pip install -r requirements.txt

# Option 2 — conda
conda env create -f environment.yml
conda activate ghana-dropout-shap
```

## Team

Oheneba Kwaku Tawiah Ntim · Jude Ahiekpor Kekeli Yao ·
Evangelina Temple · Victoria Teye · Kyei Christian Junior

Supervisor: Eric Opoku Osei (PhD), Department of Computer Science, KNUST
```

Commit message: `Update README with project structure and setup instructions`

---

**STEP 6 — Add a .gitignore to protect pupil data**

Click **"Add file" → "Create new file"**, name it `.gitignore`, paste this:
```
# Never commit raw pupil data
data-raw/
*.csv
*.xlsx
*.xls

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
*.egg-info/

# Colab / Jupyter
.virtual_documents/

# Model files (large)
*.pt
*.pth
*.pkl
```

Commit message: `Add .gitignore — protect raw pupil data from accidental commit`

---

**After all 6 steps your repository will look like this:**
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
├── federated/
├── results/
└── paper/