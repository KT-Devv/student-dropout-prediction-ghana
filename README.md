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

### Running in Colab

Two workflows are supported, and they differ in one respect that matters for
provenance.

**Working from Google Drive.** The setup cell at the top of each notebook
mounts Drive, points `DROPOUT_REPO` at the project folder and installs any
missing dependencies. This is convenient for development. It records no git
commit, so **Notebook 8 will refuse to score the held-out partition in this
mode** — see `config.require_git_commit`.

**Running from a clone.** Required for any run that scores the test
partition, because the manifest must record a real commit:

```python
!git clone https://github.com/KT-Devv/student-dropout-prediction-ghana.git
import os; os.environ["DROPOUT_REPO"] = "/content/student-dropout-prediction-ghana"
```

Paths resolve against the repository root, or against `DROPOUT_REPO` when it
is set, so the same notebooks run unchanged either way.

*Correction:* an earlier version of this README stated that no notebook calls
`drive.mount()`. All eleven do, in the Drive workflow above.

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
- De-duplication on substantive columns (identifiers and dates ignored), so
  the same pupil recorded under two study IDs cannot straddle the split.
- School-identifying variables (`school_code`, and the `geographic_zone` ×
  `school_type` pair that identifies two of the four schools) excluded from
  the primary feature set.
- Inference reported at the seed level: ten seeds re-partition the same 784
  pupils, so pooled fold intervals describe consistency across
  re-partitionings, not uncertainty about a population.
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

Pupil-level data cannot be shared publicly. Place the raw workbook at
`data-raw/`, which is git-ignored. Access to the primary field data may be
requested from the corresponding author subject to institutional ethics
approval and a data use agreement.

**Disclosure.** Between 30 June 2026 and the date this repository was made
private, the raw workbook and several processed pupil-level files were
tracked in this repository and publicly accessible. They were removed from
the full git history with `git filter-repo` and the history was force-pushed;
a fresh clone contains no pupil-level file. No forks or clones existed while
the repository was public. A Zenodo deposit containing a pseudonymised
processed dataset was deleted within the 30-day grace period. The incident is
recorded in the Limitations section of the manuscript.

## Inference boundary

Findings are scoped to pupils in Primary 4 through JHS Form 2 at **four**
basic schools in the Kumasi Metropolitan area, Ashanti Region, Ghana. The
four schools contribute unevenly and their dropout rates differ substantially;
see `results/notebook01_cleaning/*/school_cluster_audit.csv`. No claim of
generalisability beyond these four sites is made.
