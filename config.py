"""
config.py — single source of truth for this project.

Built against the ACTUAL columns in ghana_dropout_study_M.xlsx:
48 raw columns -> 44 after identifier/empty removal -> 41 in cleaned_data.csv,
1000 rows, 92 dropout cases (9.2%), 4 schools.

Every path, column name and protocol constant lives here and nowhere else.
Notebooks 1-9 import from this module; none of them redefines any of it. The
R01 submission defined the focal loss in three places, committed two SHAP
artefacts that disagreed with no label saying which model produced which, and
used two different definitions of socioeconomic_vulnerability_score in two
notebooks. Copy-paste is the defect.

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd()))
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
def find_repo_root(start=None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "config.py").exists() or (cand / ".git").exists():
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
ENGINEERED_EDA_CSV = DATA_PROCESSED / "engineered_data_FOR_EDA_ONLY.csv"

# =====================================================================
# PROTOCOL CONSTANTS
# =====================================================================
TARGET = "dropout_label"

SPLIT_SEED = 42
TEST_SIZE = 0.20
SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
N_SPLITS = 5
N_REPEATS = 5

ALPHA_LEVEL = 0.05
N_BOOT = 10_000
PRIMARY_METRIC = "auc_pr"     # must match M15. R01's Notebook 6 searched on F1.

SHARED_PARAMS = dict(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)

GAMMA_REPORTED = 2.0
ALPHA_REPORTED = 0.75

# DECLARE IN THE METHOD BEFORE READING ANY FAIRNESS OUTPUT (Q16).
FAIRNESS_THRESHOLD = 0.10

# "drop" -> school_code excluded; the claim is pupil-level WITHIN four schools
# "keep" -> retained as a predictor; the four-cluster structure then has to go
#           in the limitations
# Observed: 4 schools, WEWE alone holds 500 of the 1000 records.
SCHOOL_HANDLING = "drop"

SCORE_TEST = False
FREEZE_CONFIRMED = False

# =====================================================================
# KNOWN FACTS ABOUT THIS DATASET — assert these, don't assume them
# =====================================================================
EXPECTED = {
    "n_rows": 1000,
    "n_raw_columns": 48,
    "n_cleaned_columns": 41,        # including the target
    "n_positive": 92,               # 9.2% base rate
    "n_schools": 4,
    "academic_year": "2024/25",     # the workbook says 2024/25; M18 says
                                    # 2025/2026. One of them is wrong — fix
                                    # all four places in the manuscript.
}

# =====================================================================
# COLUMN CONFIGURATION
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

ATTENDANCE_MAX = 100.0
ATTENDANCE_WEIGHTS = np.array([0.25, 0.30, 0.45])   # term 1, 2, 3

EXAM_SCORE_COLS = ["english_exam_score", "math_exam_score",
                   "science_exam_score", "social_studies_exam_score"]
EXAM_SCORE_MAX = 100.0

# Likert 1-5 items already stored as float — no encoding needed.
LIKERT_NUMERIC_COLS = ["safety_at_home", "safety_at_school",
                       "teacher_support_rating", "school_enjoyment",
                       "class_participation"]

# Counts stored as TEXT in the workbook. Left as text they get label-encoded
# alphabetically, so "10" sorts before "2". Coerced to numbers in-fold.
NUMERIC_COERCE_COLS = [
    "grade_repetition_count",      # object, 20 distinct — a count
    "no_of_siblings_in_school",    # object, 67 distinct — a count
    "social_studies_exam_score",   # object, 202 distinct — see SUSPECT below
]

# ---------------------------------------------------------------------
# COLUMNS THAT MUST NOT BE PREDICTORS
# ---------------------------------------------------------------------
DROP_TOKENS = {"id", "serial", "index", "registration"}
DROP_EXACT = {
    "study_id", "student_id", "record_id", "pupil_id", "school_id",
    "date_recorded", "enumerator_initials",
    # --- provenance metadata, NOT a pupil attribute -------------------
    # data_source records HOW a record was collected ("Questionnaire" vs
    # "Both"), not anything about the pupil. It is nonetheless associated
    # with the outcome (r = -0.091, chi-square p = 0.007), which makes it a
    # collection artefact that a model will happily exploit. Keeping it
    # means part of the reported performance is the model learning which
    # collection route a record came through. Excluded, and stated in M6.
    "data_source",
    "notes_comments",              # 0 non-null in the workbook
}
LEAKAGE_EXACT = {
    "dropout_date", "completion_date", "graduation_date",
    "status_after_program", "final_result", "headteacher_confirmation_date",
}

# Constant in this sample; dropped in-fold. Listed so the cascade names them.
KNOWN_CONSTANT = {"academic_year", "district"}

# ---------------------------------------------------------------------
# DERIVED DUPLICATES
# ---------------------------------------------------------------------
# average_attendance is the arithmetic mean of the three term columns, all of
# which are already present. It is also the single strongest correlate of the
# outcome in the entire dataset (r = -0.782) — stronger than any exam score,
# and very close to attendance_risk_index (r = 0.783 with the outcome).
#
# The R01 pipeline dropped it via a >0.95 collinearity rule, together with
# school_feeding_status, which is an INGREDIENT of the socioeconomic
# composite. An automatic rule that deletes both the strongest predictor and
# a composite's own ingredient is why that rule is not used here.
#
# The redundancy is real, so average_attendance is dropped EXPLICITLY, for a
# stated reason. Set this to False to keep it and see what changes.
DROP_DERIVED_DUPLICATES = True
DERIVED_DUPLICATES = {
    "average_attendance": "arithmetic mean of term_1/2/3_attendance, all retained",
}

# ---------------------------------------------------------------------
# SUSPECT COLUMNS — validated in Notebook 1, never silently modelled
# ---------------------------------------------------------------------
# social_studies_exam_score: mean 105.5, SD 60.3, min 0, max 201, stored as
# TEXT with 202 distinct values, while the other three exam scores are
# numeric and bounded at 100. A score of 201 is not possible on the same
# scale. Either the paper is marked out of 200, or two fields were
# concatenated on data entry. Until somebody checks the instrument it is not
# a usable predictor. Notebook 1 reports it; this setting decides what
# happens to it:
#   "exclude" — drop it and say why (recommended until verified)
#   "rescale" — divide by 2, ONLY if the group confirms it is out of 200
#   "keep"    — keep as-is, only if the range turns out to be legitimate
SUSPECT_COLUMNS = {
    "social_studies_exam_score": {
        "issue": "values up to 201 on a 0-100 scale; stored as text; 202 distinct",
        "action": "exclude",
        "valid_range": (0.0, 100.0),
    },
}

# ---------------------------------------------------------------------
# HIGH-MISSINGNESS COLUMNS
# ---------------------------------------------------------------------
# extracurricular_activities is 24.9% missing (249 of 1000) AND is one of the
# three ingredients of behavioral_engagement_index. The R01 Notebook 3 ran
# pd.to_numeric(...).fillna(0), so 249 pupils were recorded as doing zero
# activities when the truth is that nobody asked or nobody answered. A
# quarter of that composite was fabricated. Here non-response stays NaN,
# carries its own indicator, and is imputed in-fold from the training median.
HIGH_MISSINGNESS = {
    "extracurricular_activities": 0.249,
    "daily_study_hours_at_home":  0.032,
    "no_of_siblings_in_school":   0.009,
    "class_participation":        0.002,
    "term_2_attendance":          0.001,
}
MISSINGNESS_INDICATOR_THRESHOLD = 0.05   # add a _missing flag above this rate


def drop_reason(col: str) -> str | None:
    """Why a column is dropped, or None to keep it.

    Whole-token matching, NOT substring. The R01 rule was
    `any(k in col.lower() for k in id_keywords)` with "id" in the keyword
    list, which is a substring test: any column containing those two letters
    was silently removed.
    """
    c = str(col).strip().lower()
    if c in LEAKAGE_EXACT:
        return "leakage"
    if c in DROP_EXACT:
        return "identifier / provenance metadata (exact name)"
    if DROP_DERIVED_DUPLICATES and c in DERIVED_DUPLICATES:
        return f"derived duplicate ({DERIVED_DUPLICATES[c]})"
    if c in SUSPECT_COLUMNS and SUSPECT_COLUMNS[c]["action"] == "exclude":
        return f"failed range validation ({SUSPECT_COLUMNS[c]['issue']})"
    toks = set(c.split("_"))
    hit = toks & DROP_TOKENS
    if hit:
        return f"identifier (whole token {sorted(hit)})"
    return None


def is_text(series) -> bool:
    """True for non-numeric columns.

    Do NOT use `series.dtype == object` for this. pandas 3.0 gives string
    columns a `str` dtype, so the object check silently returns False and
    every categorical is treated as numeric. Colab's pandas 2.x happens to
    work, which is luck rather than correctness.
    """
    import pandas as _pd
    return not _pd.api.types.is_numeric_dtype(series)


# =====================================================================
# CATEGORY CANONICALISATION AND ORDINAL MAPS
# =====================================================================
# Label encoding assigns integers ALPHABETICALLY. Every ordered questionnaire
# variable left out of ORDINAL_MAPS is being scrambled: alphabetically,
# travel_time_to_school orders as "1-2 hrs" < "15-30 min" < "Less than 15
# min" < "More than 2 hrs", which is meaningless as a number.
#
# Keys are lowercase and stripped; values are the canonical label.
CATEGORY_CANONICAL = {
    "family_income_level": {
        "low": "Low", "medium": "Medium", "med": "Medium",
        "high": "High", "hgh": "High",           # the observed misspelling
        "don't know": "Unknown", "dont know": "Unknown",
        "do not know": "Unknown", "unknown": "Unknown", "dk": "Unknown",
    },
}

# ---------------------------------------------------------------------
# ORDINAL MAPS  —  *** VERIFY BEFORE YOUR FIRST REAL RUN ***
# ---------------------------------------------------------------------
# Your notebook outputs gave me the DISTINCT COUNT of every categorical but
# not the full value list. Where the count below exceeds the entries in a
# map, values are missing: they become NaN and get an indicator flag.
# audit_categories() in Notebook 1 prints exactly which ones.
#
# column                            distinct   entries   status
# family_income_level                     5      3+Unk   OK
# travel_time_to_school                   6          6   VERIFY LABELS
# parent_guardian_education_level         6          6   VERIFY LABELS
# daily_study_hours_at_home               3          3   VERIFY LABELS
# own_textbooks                           7          4   INCOMPLETE
# behaviour_warnings_punishments          4          4   VERIFY LABELS
# extracurricular_activities              4          4   VERIFY LABELS
# parent_attends_school_events            6          5   INCOMPLETE
# missed_school_due_to_illness            8          3   INCOMPLETE
# missed_school_for_choreswork            6          3   INCOMPLETE
# class_level                             5          5   VERIFY LABELS
# leap_beneficiary_status                 3          3   VERIFY LABELS
# govt_support                            3          3   VERIFY LABELS
ORDINAL_MAPS = {
    "family_income_level": {"Low": 0, "Medium": 1, "High": 2},
    # "Unknown" deliberately absent -> NaN + family_income_level_missing.
    # R01 Notebook 3 mapped "don't know" to 1 (Medium), silently imputing a
    # value and hiding the non-response.

    "travel_time_to_school": {
        "Less than 15 min": 0, "15-30 min": 1, "30-45 min": 2,
        "45 min-1 hour": 3, "1-2 hrs": 4, "More than 2 hrs": 5,
    },
    "parent_guardian_education_level": {
        "None": 0, "Primary": 1, "JHS": 2, "SHS": 3, "Tertiary": 4,
        # a 6th value exists. If it is "Other" it is NOT ordinal — move the
        # column to NOMINAL_COLS instead of forcing it onto this scale.
    },
    "daily_study_hours_at_home": {
        "Less than 1 hr": 0, "1-2 hrs": 1, "More than 2 hrs": 2,
    },
    "own_textbooks": {
        "No": 0, "Yes-some subjects": 1, "Yes-most subjects": 2,
        "Yes-all subjects": 3,
    },
    "behaviour_warnings_punishments": {
        "Never": 0, "Once": 1, "Twice": 2, "More than twice": 3,
    },
    "extracurricular_activities": {
        "No activities": 0, "1 activity": 1, "2 activities": 2,
        "3 or more activities": 3,
    },
    "parent_attends_school_events": {
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4,
    },
    "missed_school_due_to_illness": {
        "No": 0, "Yes, once": 1, "Yes, more than once": 2,
    },
    "missed_school_for_choreswork": {
        "No": 0, "Yes, once": 1, "Yes, more than once": 2,
    },
    "class_level": {"P4": 0, "P5": 1, "P6": 2, "JHS1": 3, "JHS2": 4},
    "leap_beneficiary_status": {"No": 0, "Don't know": 1, "Yes": 2},
    "govt_support": {"No": 0, "Don't know": 1, "Yes": 2},
}

# Genuinely unordered — label encoding is the correct treatment for these.
NOMINAL_COLS = [
    "school_code", "geographic_zone", "school_type", "gender",
    "mode_of_transport_to_school", "parent_guardian_occupation",
    "barriers_to_regular_attendance",
]

# Free-text numeric answers appearing in count columns.
STRING_TO_NUMERIC = {
    "never": 0, "once": 1, "twice": 2, "none": 0, "no": 0, "nil": 0,
    "no activities": 0, "1 activity": 1, "2 activities": 2,
    "3 activities": 3, "more than twice": 3, "3 or more": 3,
    "3 or more activities": 3,
}


def audit_categories(df, verbose=True):
    """Print every categorical value and flag anything ORDINAL_MAPS misses.

    Run this in Notebook 1 BEFORE modelling. An unmapped category becomes
    NaN and is then imputed away, so a silent gap here loses part of a
    variable without telling anyone.
    """
    import pandas as pd
    YES_NO = {"yes", "no", "true", "false", "1", "0"}
    problems = []
    for col in df.columns:
        if col == TARGET or not is_text(df[col]):
            continue
        raw_vals = df[col].dropna().astype(str).str.strip()
        # constants are dropped in-fold; yes/no binaries are handled by the
        # yes/no detection step. Neither needs an ordinal map.
        if raw_vals.nunique() <= 1:
            continue
        if set(raw_vals.str.lower().unique()) <= YES_NO:
            if verbose:
                print(f"\n{col}  [binary yes/no]  handled by the yes/no step")
            continue
        vals = raw_vals
        canon = CATEGORY_CANONICAL.get(col)
        if canon:
            vals = vals.str.lower().map(lambda v: canon.get(v, v))
        counts = vals.value_counts()
        omap = ORDINAL_MAPS.get(col)
        if verbose:
            kind = ("ORDINAL" if omap else
                    "nominal" if col in NOMINAL_COLS else "UNCLASSIFIED")
            print(f"\n{col}  [{kind}]  {counts.size} distinct")
            for v, n in counts.items():
                mark = ""
                if omap is not None:
                    mark = f"-> {omap[v]}" if v in omap else "   *** UNMAPPED ***"
                print(f"    {str(v)[:44]:46s} {n:5d} {mark}")
        if omap is not None:
            missing = [v for v in counts.index if v not in omap]
            if missing:
                problems.append({
                    "column": col, "unmapped_values": "; ".join(map(str, missing)),
                    "n_affected": int(counts[missing].sum())})
        elif col not in NOMINAL_COLS:
            problems.append({
                "column": col, "unmapped_values": "<no ordinal map, not listed nominal>",
                "n_affected": int(len(vals))})

    if problems:
        print("\n" + "!" * 72)
        print("config.py NEEDS EDITING — ORDINAL_MAPS / NOMINAL_COLS:")
        for p in problems:
            print(f"  {p['column']}: {p['unmapped_values']} "
                  f"({p['n_affected']} records affected)")
        print("Unmapped values become NaN and are then imputed away.")
        print("!" * 72)
    else:
        print("\nEvery categorical value is accounted for.")
    return pd.DataFrame(problems)


# =====================================================================
# FEATURE FAMILIES (Q13 — family-level attribution)
# =====================================================================
FEATURE_FAMILIES = {
    "academic_performance": [
        "english_exam_score", "math_exam_score", "science_exam_score",
        "social_studies_exam_score", "average_exam_score",
        "grade_repetition_count", "class_level",
    ],
    "attendance_trajectory": [
        "term_1_attendance", "term_2_attendance", "term_3_attendance",
        "average_attendance", "attendance_risk_index",
        "missed_school_due_to_illness", "missed_school_for_choreswork",
        "barriers_to_regular_attendance",
    ],
    "household_economic": [
        "family_income_level", "leap_beneficiary_status",
        "school_feeding_status", "govt_support", "home_has_electricity",
        "own_textbooks", "no_of_siblings_in_school",
        "parent_guardian_occupation", "parent_guardian_education_level",
        "socioeconomic_vulnerability_score",
        "travel_time_to_school", "mode_of_transport_to_school",
    ],
    "behavioural_engagement": [
        "class_participation", "extracurricular_activities",
        "behaviour_warnings_punishments", "school_enjoyment",
        "teacher_support_rating", "daily_study_hours_at_home",
        "parent_attends_school_events", "behavioral_engagement_index",
    ],
    "safety_wellbeing": ["safety_at_home", "safety_at_school"],
    "demographic": ["gender", "age_at_start_of_academic_year"],
    "school_administrative": ["school_code", "school_type", "geographic_zone"],
}

_FAMILY_LOOKUP = {c: fam for fam, cols in FEATURE_FAMILIES.items() for c in cols}


def family_of(feature: str) -> str:
    f = str(feature).strip().lower()
    if f in _FAMILY_LOOKUP:
        return _FAMILY_LOOKUP[f]
    base = f[:-8] if f.endswith("_missing") else f
    if base in _FAMILY_LOOKUP:
        return _FAMILY_LOOKUP[base]
    for fam, cols in FEATURE_FAMILIES.items():
        if any(c in f for c in cols):
            return fam
    return "other_unclassified"


# =====================================================================
# RUN PROVENANCE
# =====================================================================
OUTPUT_PREFIXES = ("results/", "figures/", "models/", "data-processed/",
                   ".ipynb_checkpoints", "__pycache__")


def git_info(ignore_outputs: bool = True) -> dict:
    """Running a notebook writes results, which makes the tree dirty. If that
    counted as an uncommitted change the freeze check in Notebook 8 could
    never pass, so output paths are excluded. What must be clean is the
    SOURCE that produced the numbers."""
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a],
                                  capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    lines = [l for l in g("status", "--porcelain").splitlines() if l.strip()]
    if ignore_outputs:
        def is_output(line):
            path = line[3:].strip().strip('"')
            return any(path.startswith(p) or p in path for p in OUTPUT_PREFIXES)
        lines = [l for l in lines if not is_output(l)]
    return {"commit": g("rev-parse", "HEAD"),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(lines),
            "dirty_paths": [l[3:].strip() for l in lines][:20]}


def run_dir(tag: str) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = RESULTS / tag / run_id
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def capture_environment(out_dir: Path):
    """M19 must be pasted from this output, not retyped. The R01 M19 table
    disagreed with requirements.txt and environment.yml on every entry."""
    import pandas as pd
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=240).stdout
    except Exception as e:
        freeze = f"pip freeze failed: {e}"
    (out_dir / "pip_freeze.txt").write_text(freeze)

    keys = {"lightgbm", "scikit-learn", "shap", "pandas", "numpy", "scipy",
            "matplotlib", "seaborn", "imbalanced-learn", "ctgan", "sdv",
            "catboost", "xgboost", "joblib", "optuna", "openpyxl"}
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
            "primary_metric": PRIMARY_METRIC, "shared_params": SHARED_PARAMS,
            "gamma_reported": GAMMA_REPORTED, "alpha_reported": ALPHA_REPORTED,
            "school_handling": SCHOOL_HANDLING,
            "fairness_threshold": FAIRNESS_THRESHOLD,
            "drop_derived_duplicates": DROP_DERIVED_DUPLICATES,
            "suspect_column_actions": {k: v["action"]
                                       for k, v in SUSPECT_COLUMNS.items()},
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
    print(f"repo            : {REPO}")
    print(f"git             : {git_info()['commit'][:8] or 'no commit'}")
    print(f"school_handling : {SCHOOL_HANDLING}")
    print(f"primary metric  : {PRIMARY_METRIC}")
