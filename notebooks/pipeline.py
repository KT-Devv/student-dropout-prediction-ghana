"""
pipeline.py — the fold-safe preprocessing pipeline, the arm grid, and the
comparison machinery. Imported by Notebooks 2 through 9.

The central design point: there is no such thing as an "engineered dataset"
in this project any more. GATE-1 failed because Notebook 3 called
scaler.fit_transform() and encoder.fit_transform() on the whole frame and
wrote engineered_data.csv, which Notebook 6b then split — so every encoder,
scaler and composite had seen the held-out rows. You cannot fix that by
patching Notebook 3, because the defect is the *boundary* between Notebook 3
and Notebook 6b. Feature engineering has to happen inside the fold or it
leaks. So it lives here, as a function, and every notebook calls it.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     LeaveOneGroupOut)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

from config import *          # noqa: F401,F403
from losses import make_focal, predict_proba_focal, balanced_weights

TARGET_LABEL_MAP = {
    "0 - retained": 0, "1 - dropout": 1,
    "retained": 0, "dropout": 1, "0": 0, "1": 1,
}


# =====================================================================
# Target
# =====================================================================
def binarise_target(series: pd.Series) -> pd.Series:
    if is_text(series):
        return (series.astype(str).str.strip().str.lower()
                .map(TARGET_LABEL_MAP).astype("float"))
    return pd.to_numeric(series, errors="coerce")


# =====================================================================
# Category canonicalisation (a fixed dictionary, not a fitted statistic, so
# applying it to both partitions does not leak)
# =====================================================================
def canonicalise(series: pd.Series, colname: str) -> pd.Series:
    mapping = CATEGORY_CANONICAL.get(colname)
    if mapping is None:
        return series

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        k = str(v).strip().lower()
        if k in ("", "nan", "none", "na"):
            return np.nan
        return mapping.get(k, str(v).strip())

    return series.map(one)


def to_numeric_freetext(series: pd.Series) -> pd.Series:
    """Behaviour columns mix integers with free text ('Never', '1 activity')."""
    if not is_text(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower().replace(STRING_TO_NUMERIC)
    return pd.to_numeric(s, errors="coerce")


# =====================================================================
# THE in-fold pipeline
# =====================================================================
def preprocess_inside_fold(df_train, df_val, target_col=TARGET, verbose=False):
    """Fit every transformation on df_train, apply to df_val.

    Returns (X_train, y_train, X_val, y_val, meta).

    Every .fit()/.fit_transform() below is called on the training fold only.
    The only operations applied identically to both partitions are those that
    use fixed constants rather than data statistics: the drop rules, the
    spelling canonicalisation, the ordinal maps, and clipping attendance to
    its physical bound of 100.
    """
    train, val = df_train.copy(), df_val.copy()
    meta = {}

    # --- 1. drops, with reasons ---------------------------------------
    drops = [c for c in train.columns if c != target_col and drop_reason(c)]
    if SCHOOL_HANDLING == "drop" and SCHOOL_COL in train.columns:
        drops.append(SCHOOL_COL)
    for d in (train, val):
        d.drop(columns=[c for c in drops if c in d.columns],
               inplace=True, errors="ignore")
    meta["dropped"] = drops

    # --- 2. target ----------------------------------------------------
    for d in (train, val):
        d[target_col] = binarise_target(d[target_col])

    # --- 3. canonicalise known spelling variants ----------------------
    for col in CATEGORY_CANONICAL:
        if col in train.columns:
            for d in (train, val):
                d[col] = canonicalise(d[col], col)

    # --- 4. clip attendance to its physical bound ---------------------
    n_clipped = 0
    for col in ATTENDANCE_COLS:
        if col in train.columns:
            for d in (train, val):
                s = pd.to_numeric(d[col], errors="coerce")
                n_clipped += int((s > ATTENDANCE_MAX).sum())
                d[col] = s.clip(lower=0.0, upper=ATTENDANCE_MAX)
    meta["n_attendance_clipped"] = n_clipped

    # --- 5. free-text numerics in the behaviour columns ---------------
    for col in BEHAVIOR_COLS.values():
        if col in train.columns:
            for d in (train, val):
                d[col] = to_numeric_freetext(d[col])

    # --- 6. yes/no, detected on train ---------------------------------
    for col in list(train.columns):
        if col == target_col:
            continue
        if is_text(train[col]):
            vals = set(train[col].dropna().astype(str).str.strip().str.lower().unique())
            if vals and vals <= {"yes", "no", "true", "false", "1", "0"}:
                m = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
                for d in (train, val):
                    d[col] = (d[col].astype(str).str.strip().str.lower().map(m))

    # --- 7. explicit ordinal maps + non-response indicator ------------
    for col, mapping in ORDINAL_MAPS.items():
        if col in train.columns:
            for d in (train, val):
                mapped = d[col].map(mapping)
                d[col + "_missing"] = mapped.isna().astype(int)
                d[col] = mapped
            meta.setdefault("ordinal_applied", []).append(col)

    # --- 8. empty and constant columns, decided on train --------------
    train = train.dropna(axis=1, how="all")
    val = val[[c for c in val.columns if c in train.columns]]
    const = [c for c in train.columns
             if c != target_col and train[c].nunique(dropna=True) <= 1]
    for d in (train, val):
        d.drop(columns=[c for c in const if c in d.columns],
               inplace=True, errors="ignore")
    meta["constant_dropped"] = const

    y_train = train[target_col].copy()
    y_val = val[target_col].copy()
    X_train = train.drop(columns=[target_col])
    X_val = val.drop(columns=[target_col])

    # --- 9. imputation with TRAIN statistics --------------------------
    num = [c for c in X_train.columns if not is_text(X_train[c])]
    cat = [c for c in X_train.columns if c not in num]

    medians = X_train[num].median()
    X_train[num] = X_train[num].fillna(medians)
    X_val = X_val.reindex(columns=X_train.columns)
    X_val[num] = X_val[num].fillna(medians)

    modes = {}
    for c in cat:
        m = X_train[c].mode()
        modes[c] = m.iloc[0] if len(m) else "unknown"
        X_train[c] = X_train[c].fillna(modes[c])
        X_val[c] = X_val[c].fillna(modes[c])
    meta["n_imputed_train"] = int(df_train.isna().sum().sum())

    # --- 10. label-encode remaining nominal categoricals (fit on train) --
    encoders = {}
    for c in cat:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c].astype(str))
        known = set(le.classes_)
        X_val[c] = le.transform(
            X_val[c].astype(str).map(lambda v: v if v in known else le.classes_[0]))
        encoders[c] = le
    meta["nominal_label_encoded"] = cat

    # --- 11. composites, with every constant explicit -----------------
    att = [c for c in ATTENDANCE_COLS if c in X_train.columns]
    if len(att) == 3:
        for d in (X_train, X_val):
            frac = d[att].to_numpy(dtype=float) / ATTENDANCE_MAX
            d["attendance_risk_index"] = np.average(
                1.0 - frac, axis=1, weights=ATTENDANCE_WEIGHTS)

    leap = SOCIOECONOMIC_COLS["leap_beneficiary"]
    feed = SOCIOECONOMIC_COLS["school_feeding"]
    inc = SOCIOECONOMIC_COLS["family_income"]
    if all(c in X_train.columns for c in (leap, feed, inc)):
        # Divide by the ORDINAL maximum (2 for Low/Medium/High), a fixed
        # property of the scale — not by the training max of an alphabetically
        # label-encoded integer, which is what broke this composite in R01.
        inc_max = float(max(ORDINAL_MAPS.get(inc, {"_": 1}).values())) or 1.0
        for d in (X_train, X_val):
            vuln = 1.0 - (d[inc].astype(float) / inc_max)
            d["socioeconomic_vulnerability_score"] = (
                d[leap].astype(float) + d[feed].astype(float) + vuln) / 3.0

    warn = BEHAVIOR_COLS["behavior_warnings"]
    part = BEHAVIOR_COLS["class_participation"]
    extra = BEHAVIOR_COLS["extracurricular"]
    bcols = [warn, part, extra]
    if all(c in X_train.columns for c in bcols):
        scaler = MinMaxScaler()
        tr = scaler.fit_transform(X_train[bcols].astype(float))
        vl = scaler.transform(X_val[bcols].astype(float))
        for d, s in ((X_train, tr), (X_val, vl)):
            d["behavioral_engagement_index"] = (
                (1.0 - s[:, 0]) + s[:, 1] + s[:, 2]) / 3.0

    # --- 12. near-constant pruning on train stats, composites exempt ---
    low_var = [c for c in X_train.columns if c not in COMPOSITES
               and X_train[c].value_counts(normalize=True).iloc[0] > 0.99]
    X_train.drop(columns=low_var, inplace=True, errors="ignore")
    meta["low_variance_dropped"] = low_var

    # --- 13. align val to train exactly, column order included --------
    X_val = X_val.reindex(columns=X_train.columns)
    X_val = X_val.fillna(X_train.median(numeric_only=True))

    meta["n_features"] = X_train.shape[1]
    meta["feature_names"] = list(X_train.columns)
    meta["composites_present"] = [c for c in COMPOSITES if c in X_train.columns]
    if verbose:
        print(f"    features={meta['n_features']} "
              f"composites={len(meta['composites_present'])} "
              f"clipped={n_clipped}")
    return X_train, y_train.astype(int), X_val, y_val.astype(int), meta


def raw_feature_cols(X):
    return [c for c in X.columns if c not in COMPOSITES]


# =====================================================================
# The frozen split and the CV iterator
# =====================================================================
def frozen_split(df, target_col=TARGET):
    """The single outer 80/20 split. The test partition returned here is not
    to be scored until Notebook 8, with both freeze flags set."""
    y = binarise_target(df[target_col])
    train_pool, test_holdout = train_test_split(
        df, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    return train_pool.reset_index(drop=True), test_holdout.reset_index(drop=True)


def cv_splits(pool, seed, n_splits=N_SPLITS, n_repeats=N_REPEATS,
              target_col=TARGET):
    """Repeated stratified CV over the TRAINING POOL ONLY.

    GATE-1 (ii): Notebook 6b ran rskf.split(X, y) over all 1000 rows, so the
    200 rows reported as a held-out test partition sat inside CV training
    folds and the two evaluation modes in Table 3 were not independent.
    """
    y = binarise_target(pool[target_col]).fillna(0)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    return list(rskf.split(pool, y))


def school_splits(df, school_col=SCHOOL_COL):
    """Leave-one-school-out. With four clusters this is a bounded robustness
    check, not a generalisation estimate — and saying so is itself a finding."""
    if school_col not in df.columns:
        return []
    groups = df[school_col].astype(str).to_numpy()
    y = binarise_target(df[TARGET]).fillna(0)
    return [(tr, vl, groups[vl][0]) for tr, vl in LeaveOneGroupOut().split(df, y, groups)]


# =====================================================================
# The arm grid (GATE-2, Q20)
# =====================================================================
# Full 2x2x2: features {raw, all} x loss {ce, focal} x reweight {off, balanced}.
# Reweighting uses ONE mechanism for both losses (sample_weight), so the
# dimension is symmetric. LEGACY_A reproduces the R01 baseline definition
# (is_unbalance=True, raw features) purely so the new run reconciles against
# the old tables.
ARMS = {
    "A_raw_ce_noW":    dict(features="raw", loss="ce",    reweight=False, legacy=False),
    "B_raw_ce_W":      dict(features="raw", loss="ce",    reweight=True,  legacy=False),
    "C_raw_focal_noW": dict(features="raw", loss="focal", reweight=False, legacy=False),
    "D_raw_focal_W":   dict(features="raw", loss="focal", reweight=True,  legacy=False),
    "E_all_ce_noW":    dict(features="all", loss="ce",    reweight=False, legacy=False),
    "F_all_ce_W":      dict(features="all", loss="ce",    reweight=True,  legacy=False),
    "G_all_focal_noW": dict(features="all", loss="focal", reweight=False, legacy=False),
    "H_all_focal_W":   dict(features="all", loss="focal", reweight=True,  legacy=False),
    "LEGACY_A":        dict(features="raw", loss="ce",    reweight=False, legacy=True),
}

# Identical feature matrix, identical weighting, one argument different.
HEADLINE = ("E_all_ce_noW", "G_all_focal_noW")

# What the R01 manuscript reported as the headline: three changes at once.
OLD_HEADLINE = ("LEGACY_A", "G_all_focal_noW")

ARM_LABELS = {
    "A_raw_ce_noW":    "Baseline LightGBM (raw, cross-entropy)",
    "B_raw_ce_W":      "Baseline + class weighting",
    "C_raw_focal_noW": "Focal loss on raw features  [the cell R01 never ran]",
    "D_raw_focal_W":   "Focal loss on raw features + weighting",
    "E_all_ce_noW":    "Composites, cross-entropy   [matched reference]",
    "F_all_ce_W":      "Composites, cross-entropy + weighting",
    "G_all_focal_noW": "E-LightGBM (composites, focal loss)",
    "H_all_focal_W":   "E-LightGBM + weighting",
    "LEGACY_A":        "R01 baseline as published (is_unbalance=True)",
}


def fit_arm(name, X_tr, y_tr, seed=SPLIT_SEED, gamma=GAMMA_REPORTED,
            alpha=ALPHA_REPORTED):
    """Fit one arm. Returns (model, predict_fn, columns_used)."""
    from lightgbm import LGBMClassifier

    spec = ARMS[name]
    cols = raw_feature_cols(X_tr) if spec["features"] == "raw" else list(X_tr.columns)
    sw = balanced_weights(y_tr) if spec["reweight"] else None
    kw = dict(random_state=seed, **SHARED_PARAMS)

    if spec["loss"] == "ce":
        model = LGBMClassifier(objective="binary",
                               is_unbalance=bool(spec["legacy"]), **kw)
        model.fit(X_tr[cols], y_tr, sample_weight=sw)

        def predict(X_eval):
            return model.predict_proba(X_eval[cols])[:, 1]
    else:
        obj, ev = make_focal(gamma, alpha)
        model = LGBMClassifier(objective=obj, **kw)
        model.fit(X_tr[cols], y_tr, sample_weight=sw, eval_metric=ev)

        def predict(X_eval):
            return predict_proba_focal(model, X_eval, cols)

    return model, predict, cols


# =====================================================================
# Model persistence that does not depend on pickling the objective
# =====================================================================
def save_model_bundle(model, columns, arm, path, gamma=GAMMA_REPORTED,
                      alpha=ALPHA_REPORTED):
    """Persist a fitted arm WITHOUT serialising its objective function.

    A LightGBM booster is saved as a text string, which carries the trained
    trees and nothing else. The objective was only ever needed during
    training, so it does not have to survive the round trip — and trying to
    make it survive is what produced the R01 Notebook 8 workaround
    (redefining focal_loss_lgb locally so pickle could resolve
    __main__.focal_loss_lgb).
    """
    import joblib
    booster = getattr(model, "booster_", None)
    payload = {
        "arm": arm, "columns": list(columns), "spec": ARMS[arm],
        "gamma": gamma, "alpha": alpha,
        "shared_params": SHARED_PARAMS,
        "booster_string": booster.model_to_string() if booster is not None else None,
    }
    if payload["booster_string"] is None:
        # non-LightGBM estimator (or a stub): fall back to a plain pickle
        payload["sklearn_model"] = model
    joblib.dump(payload, path)
    return payload


def load_model_bundle(path):
    """Return (predict_proba_fn, columns, payload). Needs no objective."""
    import joblib
    payload = joblib.load(path)
    cols = payload["columns"]
    if payload.get("booster_string"):
        import lightgbm as lgb
        booster = lgb.Booster(model_str=payload["booster_string"])
        is_focal = payload["spec"]["loss"] == "focal"

        def predict(X_eval):
            raw = booster.predict(X_eval[cols], raw_score=is_focal)
            return 1.0 / (1.0 + np.exp(-raw)) if is_focal else raw
    else:
        m = payload["sklearn_model"]

        def predict(X_eval):
            return m.predict_proba(X_eval[cols])[:, 1]
    return predict, cols, payload


# =====================================================================
# Metrics
# =====================================================================
def score_binary(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true)
    yhat = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, yhat),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "macro_f1": f1_score(y_true, yhat, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(y_true, p) if len(set(y_true)) > 1 else np.nan,
        "auc_pr": average_precision_score(y_true, p),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "n": int(len(y_true)), "n_positive": int((y_true == 1).sum()),
        "base_rate_pct": round(100 * float((y_true == 1).mean()), 2),
        "pred_std": float(np.std(p)), "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)), "pred_mean": float(np.mean(p)),
    }


# =====================================================================
# Grid runner
# =====================================================================
def run_grid(pool, seeds=SEEDS, arms=tuple(ARMS), gamma=GAMMA_REPORTED,
             alpha=ALPHA_REPORTED, collect_predictions=False, verbose=True):
    """Every arm, every fold, every seed — on the training pool only.

    Returns (fold_scores_df, predictions_df_or_None).
    """
    rows, pred_frames = [], []
    t0_all = time.perf_counter()
    for si, seed in enumerate(seeds, 1):
        for fi, (tr, vl) in enumerate(cv_splits(pool, seed), 1):
            X_tr, y_tr, X_vl, y_vl, meta = preprocess_inside_fold(
                pool.iloc[tr], pool.iloc[vl])
            for arm in arms:
                t0 = time.perf_counter()
                _, predict, cols = fit_arm(arm, X_tr, y_tr, seed, gamma, alpha)
                fit_s = time.perf_counter() - t0
                p = predict(X_vl)
                rows.append({"seed": seed, "fold": fi, "arm": arm,
                             "n_features": len(cols),
                             "fit_seconds": fit_s,
                             "gamma": gamma, "alpha": alpha,
                             **score_binary(y_vl, p)})
                if collect_predictions:
                    pred_frames.append(pd.DataFrame({
                        "seed": seed, "fold": fi, "arm": arm,
                        "row_id": pool.index[vl],
                        "y": y_vl.to_numpy(), "p": p}))
        if verbose:
            print(f"  seed {seed} ({si}/{len(seeds)}) "
                  f"[{time.perf_counter()-t0_all:.0f}s]")
    fold_df = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else None
    return fold_df, preds


# =====================================================================
# Variance and paired comparison
# =====================================================================
def two_level_variance(fold_df, metric=PRIMARY_METRIC):
    """Between-seed SD and within-seed fold SD, from the same pipeline.

    J3 requires both, separately AND comparably. R01 reported only the
    within-seed fold SD and described it as variance across 25 folds — which
    reads as 25 independent observations but is five repeats of one
    partitioning scheme at one seed.
    """
    per_seed = (fold_df.groupby(["arm", "seed"])[metric]
                .agg(seed_mean="mean", within_seed_fold_sd="std").reset_index())
    out = (per_seed.groupby("arm")
           .agg(mean=("seed_mean", "mean"),
                between_seed_sd=("seed_mean", "std"),
                mean_within_seed_fold_sd=("within_seed_fold_sd", "mean"))
           .reset_index().sort_values("mean", ascending=False))
    out["n_seeds"] = fold_df["seed"].nunique()
    out["n_folds_per_seed"] = fold_df.groupby(["arm", "seed"]).size().max()
    out["label"] = out["arm"].map(ARM_LABELS)
    return out


def bootstrap_ci(diffs, n_boot=N_BOOT, level=0.95, seed=SPLIT_SEED):
    """Percentile bootstrap CI on a mean paired difference.

    R01 reported a 95% CI of (-0.007, 0.009) in both R2 and R4 with no
    computation behind it in any committed notebook. This is the computation.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def compare_arms(fold_df, ref_arm, test_arm, label="", metric=PRIMARY_METRIC):
    """Paired comparison of test_arm minus ref_arm, per seed and overall."""
    from scipy.stats import wilcoxon

    label = label or f"{test_arm} - {ref_arm}"
    wide = fold_df.pivot_table(index=["seed", "fold"], columns="arm", values=metric)
    if ref_arm not in wide or test_arm not in wide:
        raise KeyError(f"missing arm in fold_df: {ref_arm} / {test_arm}")

    per_seed = []
    for seed, sub in wide.groupby(level="seed"):
        d = (sub[test_arm] - sub[ref_arm]).dropna().to_numpy()
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except Exception:
            p = np.nan
        pooled = np.sqrt((sub[ref_arm].std(ddof=1) ** 2
                          + sub[test_arm].std(ddof=1) ** 2) / 2)
        lo, hi = bootstrap_ci(d)
        per_seed.append({
            "contrast": label, "seed": seed,
            "ref_mean": sub[ref_arm].mean(), "test_mean": sub[test_arm].mean(),
            "diff_mean": d.mean(), "sign": "+" if d.mean() > 0 else "-",
            "wilcoxon_p": p,
            "cohens_d": d.mean() / pooled if pooled > 0 else np.nan,
            "ci95_lo": lo, "ci95_hi": hi, "n_folds": len(d),
        })
    per_seed = pd.DataFrame(per_seed)

    all_d = (wide[test_arm] - wide[ref_arm]).dropna().to_numpy()
    lo, hi = bootstrap_ci(all_d)
    grand = per_seed["diff_mean"].mean()
    sd = per_seed["diff_mean"].std(ddof=1)
    summary = {
        "contrast": label, "ref_arm": ref_arm, "test_arm": test_arm,
        "grand_mean_diff": grand, "between_seed_sd": sd,
        "ci95_lo": lo, "ci95_hi": hi,
        "n_seeds_favouring_test": int((per_seed["sign"] == "+").sum()),
        "n_seeds_favouring_ref": int((per_seed["sign"] == "-").sum()),
        "n_seeds_p_below_alpha": int((per_seed["wilcoxon_p"] < ALPHA_LEVEL).sum()),
        "sd_exceeds_effect": bool(sd >= abs(grand)) if np.isfinite(sd) else None,
        "n_seeds": len(per_seed),
    }
    return per_seed, summary


# =====================================================================
# Threshold to caseload (Q12)
# =====================================================================
def caseload_table(preds, arm, thresholds=(0.10, 0.19, 0.20, 0.30, 0.40,
                                           0.50, 0.60, 0.70, 0.80)):
    """Convert a ranking metric into the two numbers a headteacher needs:
    how many pupils per hundred get flagged, and how many real dropouts
    that catches."""
    sub = preds[preds["arm"] == arm]
    rows = []
    for thr in thresholds:
        per_fold = []
        for _, g in sub.groupby(["seed", "fold"]):
            y = g["y"].to_numpy()
            yhat = (g["p"].to_numpy() >= thr).astype(int)
            n = len(y)
            tp = int(((yhat == 1) & (y == 1)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())
            per_fold.append({
                "flagged_per_100": 100 * (tp + fp) / n,
                "precision_among_flagged": tp / (tp + fp) if (tp + fp) else np.nan,
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "missed_dropouts_per_100": 100 * fn / n,
            })
        rows.append({"threshold": thr,
                     **pd.DataFrame(per_fold).mean().round(4).to_dict()})
    out = pd.DataFrame(rows)
    out["arm"] = arm
    return out


def fixed_capacity_recall(preds, arm, ks=(5, 10, 15, 20)):
    """A headteacher who can review only k pupils per 100: what fraction of
    the pupils who actually left are in that list?"""
    sub = preds[preds["arm"] == arm]
    rows = []
    for k in ks:
        vals = []
        for _, g in sub.groupby(["seed", "fold"]):
            gg = g.sort_values("p", ascending=False)
            take = max(1, int(round(k / 100 * len(gg))))
            denom = max(int(gg["y"].sum()), 1)
            vals.append(gg.head(take)["y"].sum() / denom)
        rows.append({"reviewed_per_100": k, "arm": arm,
                     "recall_at_capacity": float(np.mean(vals))})
    return pd.DataFrame(rows)
