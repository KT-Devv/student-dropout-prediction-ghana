"""
config.py — single source of truth for this project.

Every path, column name and protocol constant lives here and nowhere else.
Notebooks 1 through 9 import from this module; none of them redefines any
of it locally. That is deliberate: the R01 submission had the focal loss
defined in three places, two disagreeing SHAP artefacts, and two different
socioeconomic composites (Notebook 3 mapped income ordinally, Notebook 9
label-encoded it alphabetically), which is why the two seed-42 rows
disagreed. Import, do not copy.

Usage in a notebook:

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd()))   # or the repo root
    from config import *
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
import json
from pathlib import Path

import numpy as np

# =====================================================================
# PATHS — resolved against the repository root, never hard-coded to Drive
# =====================================================================
# Q4 fix. Every R01 notebook hard-coded
# PROJECT_DIR = "/content/drive/MyDrive/Ghana_Dropout_Project" and called
# drive.mount(), so no notebook ran against a fresh clone. Set the
# DROPOUT_REPO environment variable to override the search below.


def find_repo_root(start: str | os.PathLike | None = None) -> Path:
    """Walk upward looking for a repository marker."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists() or (cand / "requirements.txt").exists():
            return cand
    return p


REPO = Path(os.environ["DROPOUT_REPO"]) if os.environ.get("DROPOUT_REPO") else find_repo_root()

DATA_RAW       = REPO / "data-raw"
DATA_PROCESSED = REPO / "data-processed"
RESULTS        = REPO / "results"
FIGURES        = REPO / "figures"
MODELS         = REPO / "models"

for _d in (DATA_PROCESSED, RESULTS, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)


def first_existing(*relative_paths) -> Path | None:
    for r in relative_paths:
        q = REPO / r
        if q.exists():
            return q
    return None


RAW_WORKBOOK = first_existing(
    "data-raw/ghana_dropout_study_M.xlsx",
    "ghana_dropout_study_M.xlsx",
    "data/ghana_dropout_study_M.xlsx",
)
CLEANED_CSV = DATA_PROCESSED / "cleaned_data.csv"

# Illustrative only. NOT an input to any model — see Notebook 3.
ENGINEERED_EDA_CSV = DATA_PROCESSED / "engineered_data_FOR_EDA_ONLY.csv"

# =====================================================================
# PROTOCOL CONSTANTS
# =====================================================================
TARGET = "dropout_label"

SPLIT_SEED = 42      # the frozen outer 80/20 split. Never varied.
TEST_SIZE = 0.20

SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
N_SPLITS = 5
N_REPEATS = 5

ALPHA_LEVEL = 0.05   # significance level
N_BOOT = 10_000      # bootstrap resamples for paired CIs

PRIMARY_METRIC = "auc_pr"   # must match M15. The R01 Notebook 6 searched on
                            # F1 while the manuscript declared AUC-PR primary.

SHARED_PARAMS = dict(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
)

GAMMA_REPORTED = 2.0
ALPHA_REPORTED = 0.75

# DECLARE THIS IN THE METHOD BEFORE READING ANY FAIRNESS OUTPUT (Q16).
FAIRNESS_THRESHOLD = 0.10

# "drop" -> school_code excluded; the claim is pupil-level WITHIN four schools
# "keep" -> school_code retained as a predictor; four-cluster structure must
#           then be declared in the limitations
SCHOOL_HANDLING = "drop"

# Gate on test-set scoring. See Notebook 8.
SCORE_TEST = False
FREEZE_CONFIRMED = False

# =====================================================================
# COLUMN CONFIGURATION — fix names here on the first pass, nowhere else
# =====================================================================
ATTENDANCE_COLS = ["term_1_attendance", "term_2_attendance", "term_3_attendance"]

SOCIOECONOMIC_COLS = {
    "leap_beneficiary": "leap_beneficiary_status",
    "school_feeding":   "school_feeding_status",
    "family_income":    "family_income_level",
}

BEHAVIOR_COLS = {
    "behavior_warnings":   "behaviour_warnings_punishments",
    "class_participation": "class_participation",
    "extracurricular":     "extracurricular_activities",
}

SCHOOL_COL = "school_code"
GENDER_COL = "gender"
GEO_COL    = "geographic_zone"

COMPOSITES = [
    "attendance_risk_index",
    "socioeconomic_vulnerability_score",
    "behavioral_engagement_index",
]

ATTENDANCE_MAX = 100.0   # physical bound; a fixed constant, not a data statistic
ATTENDANCE_WEIGHTS = np.array([0.25, 0.30, 0.45])   # term 1, 2, 3

# ---------------------------------------------------------------------
# Identifier and leakage removal — WHOLE TOKENS, not substrings
# ---------------------------------------------------------------------
# The R01 rule was `any(k in col.lower() for k in id_keywords)` with "id" in
# the keyword list. That is a substring test: "residence_type",
# "resident_status" and "provider_of_uniform" all contain the letters "id"
# and were silently removed. This is a live candidate explanation for the
# unreconciled 44 -> 41 column path (Q2, Q11).
DROP_TOKENS = {"id", "serial", "index", "registration"}
DROP_EXACT = {
    "student_id", "study_id", "record_id", "pupil_id", "school_id",
    "date_recorded", "enumerator_initials",
}
LEAKAGE_EXACT = {
    "dropout_date", "completion_date", "graduation_date",
    "status_after_program", "final_result", "headteacher_confirmation_date",
}


def is_text(series) -> bool:
    """True for non-numeric (categorical/text) columns.

    Do NOT use `series.dtype == object` for this. pandas 3.0 gives string
    columns a `str` dtype rather than `object`, so the object check silently
    returns False and every categorical column gets treated as numeric —
    which makes label encoding, mode imputation and the yes/no detection all
    no-ops. Colab currently ships pandas 2.x where the object check happens
    to work, but that is luck, not correctness.
    """
    import pandas as _pd
    return not _pd.api.types.is_numeric_dtype(series)


def drop_reason(col: str) -> str | None:
    """Why a column is dropped, or None to keep it.

    Precedence is leakage > identifier, so a column that is both is reported
    as leakage — the reason a reader needs should not be hidden behind an
    accident of ordering.
    """
    c = str(col).strip().lower()
    if c in LEAKAGE_EXACT:
        return "leakage"
    if c in DROP_EXACT:
        return "identifier (exact name)"
    toks = set(c.split("_"))
    hit = toks & DROP_TOKENS
    if hit:
        return f"identifier (whole token {sorted(hit)})"
    return None


# ---------------------------------------------------------------------
# Category canonicalisation and ordinal maps (Q3, Q14)
# ---------------------------------------------------------------------
# Label encoding assigns integers alphabetically. On this data
# family_income_level contains both "High" and a misspelled "Hgh", so
# alphabetical encoding gave the order Don't know, High, Hgh, Low, Medium.
# The socioeconomic composite was therefore not monotone in income and was
# partly a function of a typo. Notebook 3 happened to map income ordinally;
# Notebook 9 did not. Same-named feature, two different definitions, two
# disagreeing sets of results. Fixed here, once.
CATEGORY_CANONICAL = {
    "family_income_level": {
        "hgh": "High", "high": "High",
        "med": "Medium", "medium": "Medium",
        "low": "Low",
        "don't know": "Unknown", "dont know": "Unknown",
        "do not know": "Unknown", "unknown": "Unknown", "dk": "Unknown",
    },
}

# Genuinely ordered variables. ANY ordered variable left out of this dict is
# still being label-encoded alphabetically — go through the codebook and add
# them (parental education band, travel distance band, text Likert items).
ORDINAL_MAPS = {
    "family_income_level": {"Low": 0, "Medium": 1, "High": 2},
    # "Unknown" is deliberately absent: it maps to NaN and gets a separate
    # <col>_missing indicator. The R01 Notebook 3 mapped "don't know" to 1
    # (Medium), which silently imputes a value and hides the non-response.
    # TODO: add your remaining ordered questionnaire variables here.
}

# Free-text numeric answers seen in the behaviour columns
STRING_TO_NUMERIC = {
    "never": 0, "once": 1, "twice": 2, "no activities": 0,
    "1 activity": 1, "2 activities": 2, "3 activities": 3, "none": 0,
}

# ---------------------------------------------------------------------
# Feature families for the Q13 family-level attribution
# ---------------------------------------------------------------------
FEATURE_FAMILIES = {
    "academic_performance":   ["exam_score", "average_exam", "grade_repetition",
                               "repeat", "term_grade", "score"],
    "attendance_trajectory":  ["attendance", "absent"],
    "household_economic":     ["income", "leap", "feeding", "household",
                               "guardian", "parent", "sibling", "work",
                               "distance", "travel"],
    "behavioural_engagement": ["participation", "extracurricular", "warning",
                               "punish", "enjoy", "teacher_support",
                               "engagement", "discipline"],
    "school_administrative":  ["school_code", "school_type", "class_size"],
}


def family_of(feature: str) -> str:
    f = str(feature).lower()
    for fam, keys in FEATURE_FAMILIES.items():
        if any(k in f for k in keys):
            return fam
    return "other_unclassified"


# =====================================================================
# RUN PROVENANCE — one locked run identifier per execution (GATE-3)
# =====================================================================
# Paths whose contents do not make the tree "dirty" for freeze purposes.
# Writing a results directory is the normal consequence of running a
# notebook; if that counted as an uncommitted change, the freeze check in
# Notebook 8 could never pass. What must be clean is the SOURCE that
# produced the numbers.
OUTPUT_PREFIXES = ("results/", "figures/", "models/", "data-processed/",
                   ".ipynb_checkpoints", "__pycache__")


def git_info(ignore_outputs: bool = True) -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a],
                                  capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    porcelain = g("status", "--porcelain")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    if ignore_outputs:
        def is_output(line):
            path = line[3:].strip().strip('"')
            return any(path.startswith(p) or p in path for p in OUTPUT_PREFIXES)
        lines = [l for l in lines if not is_output(l)]
    return {"commit": g("rev-parse", "HEAD"),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(lines),
            "dirty_paths": [l[3:].strip() for l in lines][:20],
            "source_clean_definition": "outputs excluded: " + ", ".join(OUTPUT_PREFIXES)}


def run_dir(tag: str) -> Path:
    """Create and return results/<tag>/<RUN_ID>/ for this execution."""
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = RESULTS / tag / run_id
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def capture_environment(out_dir: Path):
    """Write pip freeze and a trimmed version table. M19 must be pasted from
    this output, not retyped — the R01 M19 table disagreed with both
    requirements.txt and environment.yml on every single entry."""
    import pandas as pd
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=240).stdout
    except Exception as e:
        freeze = f"pip freeze failed: {e}"
    (out_dir / "pip_freeze.txt").write_text(freeze)

    keys = {"lightgbm", "scikit-learn", "shap", "pandas", "numpy", "scipy",
            "matplotlib", "seaborn", "imbalanced-learn", "ctgan", "sdv",
            "catboost", "xgboost", "joblib", "optuna"}
    rows = [{"package": "python", "version": platform.python_version()}]
    for line in freeze.splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            if name.strip().lower() in keys:
                rows.append({"package": name.strip(), "version": ver.strip()})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / "environment_versions.csv", index=False)
    return tbl


def write_manifest(out_dir: Path, extra: dict | None = None) -> dict:
    manifest = {
        "run_dir": str(out_dir.relative_to(REPO)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "protocol": {
            "split_seed": SPLIT_SEED, "test_size": TEST_SIZE, "seeds": SEEDS,
            "n_splits": N_SPLITS, "n_repeats": N_REPEATS,
            "primary_metric": PRIMARY_METRIC,
            "shared_params": SHARED_PARAMS,
            "gamma_reported": GAMMA_REPORTED, "alpha_reported": ALPHA_REPORTED,
            "school_handling": SCHOOL_HANDLING,
            "fairness_threshold": FAIRNESS_THRESHOLD,
        },
    }
    if extra:
        manifest.update(extra)
    (out_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def banner(title: str):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"repo      : {REPO}")
    print(f"git       : {git_info()}")
