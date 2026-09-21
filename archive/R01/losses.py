"""
losses.py — the focal-loss objective, defined exactly once.

Changes from the R01 version:

1. `make_focal(gamma, alpha)` is a factory, so gamma and alpha can be swept
   (Q8: the R01 submission fixed them at 2.0 and 0.75 on the authority of
   Lin et al. and never varied them; a null at one point in a two-parameter
   space is not a null on the method).
2. `focal_loss_lgb` / `focal_loss_eval` are kept at the reported parameters
   so old pickles still resolve their `objective` reference.
3. `balanced_weights` lives here too, because class reweighting must use the
   SAME mechanism in both arms. `is_unbalance=True` is honoured only by
   LightGBM's built-in objectives and is inert under a custom objective, so
   the R01 comparison gave the baseline an explicit weighting correction and
   the proposed arm none beyond the focal alpha term. That is a fourth
   asymmetry on top of the three the examination named.

The argument-order requirement below is the defect the group found and
documented, and it must not reappear: LightGBM's sklearn API calls a custom
objective positionally as objective(y_true, y_pred), with both arguments as
plain numpy arrays. Do NOT call .get_label() on either — that convention
belongs to the native lgb.train() API. Reversing the order silently swaps
labels and raw scores and collapses the model to the majority class,
observable as AUC-ROC = 0.500 and zero minority recall.
"""

from __future__ import annotations

import numpy as np

GAMMA = 2.0        # focusing exponent; downweights easy, well-classified cases
ALPHA = 0.75       # class-weighting scalar; upweights the minority class
EPSILON = 1e-6


def make_focal(gamma: float = GAMMA, alpha: float = ALPHA, eps: float = EPSILON):
    """Return (objective, eval_metric) for binary focal loss (Lin et al., 2017).

    objective(y_true, y_pred) -> (grad, hess)
    eval_metric(y_true, y_pred) -> (name, value, is_higher_better)

    y_pred is the raw margin (logit), not a probability.
    """
    def focal_objective(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        p = np.clip(1.0 / (1.0 + np.exp(-y_pred)), eps, 1.0 - eps)
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        w = alpha_t * (1.0 - pt) ** gamma
        grad = w * (p - y_true)
        hess = w * p * (1.0 - p)
        return grad, hess

    def focal_eval(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        p = np.clip(1.0 / (1.0 + np.exp(-y_pred)), eps, 1.0 - eps)
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        loss = -alpha_t * (1.0 - pt) ** gamma * np.log(pt)
        return "focal_loss", float(np.mean(loss)), False

    focal_objective.gamma = gamma
    focal_objective.alpha = alpha
    return focal_objective, focal_eval


# ---------------------------------------------------------------------
# Top-level (NOT closure) bindings at the reported parameterisation.
# ---------------------------------------------------------------------
# These must be defined as real module-level functions, not as the return
# value of make_focal(). Pickle serialises a function by qualified name, and
# a closure's qualname is "make_focal.<locals>.focal_objective", which does
# not resolve on load:
#
#   PicklingError: Can't pickle <function make_focal.<locals>.focal_objective>:
#   it's not found as losses.make_focal.<locals>.focal_objective
#
# That is the same failure the R01 Notebook 8 worked around by redefining
# focal_loss_lgb in its own __main__ so pickle could resolve the reference.
# The workaround is what let the two copies drift apart. Defining the
# functions properly here removes the need for it.


def focal_loss_lgb(y_true, y_pred):
    """Binary focal loss grad/hess at GAMMA and ALPHA. Picklable by name."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(1.0 / (1.0 + np.exp(-y_pred)), EPSILON, 1.0 - EPSILON)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, ALPHA, 1.0 - ALPHA)
    w = alpha_t * (1.0 - pt) ** GAMMA
    return w * (p - y_true), w * p * (1.0 - p)


def focal_loss_eval(y_true, y_pred):
    """Focal loss value at GAMMA and ALPHA. Picklable by name."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(1.0 / (1.0 + np.exp(-y_pred)), EPSILON, 1.0 - EPSILON)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, ALPHA, 1.0 - ALPHA)
    loss = -alpha_t * (1.0 - pt) ** GAMMA * np.log(pt)
    return "focal_loss", float(np.mean(loss)), False


_make_focal_closure = make_focal


def make_focal(gamma: float = GAMMA, alpha: float = ALPHA, eps: float = EPSILON):
    """As above, but returns the PICKLABLE top-level pair when the parameters
    are the reported ones, so a model fitted at (GAMMA, ALPHA) can be saved.
    Swept parameterisations still get closures — those models are not
    picklable and should be saved with pipeline.save_model_bundle()."""
    if (gamma, alpha, eps) == (GAMMA, ALPHA, EPSILON):
        return focal_loss_lgb, focal_loss_eval
    return _make_focal_closure(gamma, alpha, eps)


def predict_proba_focal(model, X_eval, columns=None):
    """Probabilities from a model trained with a custom objective.

    LightGBM's .predict_proba() does NOT work correctly under a custom
    objective — it returns raw margin scores, not calibrated probabilities.
    Always route through here.
    """
    X = X_eval[columns] if columns is not None else X_eval
    raw = model.predict(X, raw_score=True)
    return 1.0 / (1.0 + np.exp(-raw))


def balanced_weights(y):
    """Per-instance weights equivalent to is_unbalance=True, but applied via
    sample_weight so the SAME mechanism works under both the built-in and the
    custom objective. This is what makes the reweighting dimension of the
    ablation grid symmetric across arms."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n_neg / n_pos, 1.0).astype(float)
