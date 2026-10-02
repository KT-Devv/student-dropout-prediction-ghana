"""
losses.py — the focal-loss objective, defined exactly once.

R03 CHANGE (verification finding N5)
------------------------------------
The objective now takes THREE arguments, (y_true, y_pred, sample_weight),
and multiplies the gradient and Hessian by the weight.

LightGBM's sklearn API inspects the number of parameters of a custom
objective and dispatches accordingly:

    2 parameters -> func(labels, preds)                  <- weights NEVER passed
    3 parameters -> func(labels, preds, weight)
    4 parameters -> func(labels, preds, weight, group)

The R02 objective had two parameters, so `sample_weight` supplied to .fit()
was silently discarded in every focal arm. Arms D and H were therefore
identical to C and G to machine precision, the 2x2x2 grid contained six
distinct models rather than eight, and the reweighting asymmetry that the
is_unbalance fix was meant to close was reintroduced by another route.

LightGBM does not re-apply the weights after calling a custom objective, so
the multiplication below is the only place they take effect.

ARGUMENT ORDER (unchanged requirement)
--------------------------------------
Arguments are positional: (y_true, y_pred, weight). Do NOT call .get_label()
on any of them — that convention belongs to the native lgb.train() API, not
the sklearn wrapper. Reversing y_true and y_pred silently swaps labels and
raw scores and collapses the model to the majority class, observable as
AUC-ROC = 0.500 and zero minority recall.
"""

from __future__ import annotations

import numpy as np

GAMMA = 2.0        # focusing exponent; downweights easy, well-classified cases
ALPHA = 0.75       # class-weighting scalar; upweights the minority class
EPSILON = 1e-6


def _focal_grad_hess(y_true, y_pred, weight, gamma, alpha, eps):
    """Gradient and Hessian of binary focal loss, scaled by instance weight."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(1.0 / (1.0 + np.exp(-np.asarray(y_pred, dtype=float))), eps, 1.0 - eps)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    focal_w = alpha_t * (1.0 - pt) ** gamma

    grad = focal_w * (p - y_true)
    hess = focal_w * p * (1.0 - p)

    if weight is not None:
        w = np.asarray(weight, dtype=float)
        grad = grad * w
        hess = hess * w
    return grad, hess


def _focal_value(y_true, y_pred, gamma, alpha, eps):
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(1.0 / (1.0 + np.exp(-np.asarray(y_pred, dtype=float))), eps, 1.0 - eps)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    return float(np.mean(-alpha_t * (1.0 - pt) ** gamma * np.log(pt)))


def make_focal(gamma: float = GAMMA, alpha: float = ALPHA, eps: float = EPSILON):
    """Return (objective, eval_metric) for binary focal loss (Lin et al., 2017).

    objective(y_true, y_pred, weight) -> (grad, hess)     [three arguments]
    eval_metric(y_true, y_pred)       -> (name, value, is_higher_better)

    y_pred is the raw margin (logit), not a probability.

    At the reported parameterisation the PICKLABLE module-level functions are
    returned instead of closures, so a fitted model can be saved. Swept
    parameterisations return closures; save those through
    pipeline.save_model_bundle(), which stores the booster without the
    objective.
    """
    if (gamma, alpha, eps) == (GAMMA, ALPHA, EPSILON):
        return focal_loss_lgb, focal_loss_eval

    def focal_objective(y_true, y_pred, weight):
        return _focal_grad_hess(y_true, y_pred, weight, gamma, alpha, eps)

    def focal_eval(y_true, y_pred):
        return "focal_loss", _focal_value(y_true, y_pred, gamma, alpha, eps), False

    focal_objective.gamma = gamma
    focal_objective.alpha = alpha
    return focal_objective, focal_eval


# ---------------------------------------------------------------------
# Top-level (NOT closure) bindings at the reported parameterisation.
# ---------------------------------------------------------------------
# These must be real module-level functions. Pickle serialises a function by
# qualified name, and a closure's qualname is
# "make_focal.<locals>.focal_objective", which does not resolve on load. That
# is the failure the R01 Notebook 8 worked around by redefining the objective
# in its own __main__, which is how three copies of it drifted apart.

def focal_loss_lgb(y_true, y_pred, weight):
    """Binary focal loss grad/hess at GAMMA and ALPHA, weight-aware.

    THREE arguments: LightGBM passes sample weights only to a three-argument
    custom objective. See the module docstring.
    """
    return _focal_grad_hess(y_true, y_pred, weight, GAMMA, ALPHA, EPSILON)


def focal_loss_eval(y_true, y_pred):
    """Focal loss value at GAMMA and ALPHA. Picklable by name."""
    return "focal_loss", _focal_value(y_true, y_pred, GAMMA, ALPHA, EPSILON), False


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
    """Per-instance weights equivalent to is_unbalance=True, applied through
    sample_weight so the SAME mechanism works under both the built-in and the
    custom objective. This is what makes the reweighting dimension of the
    ablation grid symmetric across arms — and, from R03, it actually takes
    effect under the focal objective."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n_neg / n_pos, 1.0).astype(float)


def objective_arity(func) -> int:
    """Number of parameters LightGBM dispatches on. Used by the
    weight-effectiveness assertion in pipeline.assert_weighting_effective()."""
    from inspect import signature
    return len(signature(func).parameters)


# =====================================================================
# EXACT focal gradient — R03 addition
# =====================================================================
# FINDING (not raised in the R02 verification):
#
# The gradient above treats the focal modulating factor (1 - pt)^gamma as a
# constant instead of differentiating it. It is therefore not the derivative
# of the focal loss. Verified numerically: the implemented gradient differs
# from the central-difference derivative of the loss by up to 0.31 in raw
# margin units (correlation 0.978, sign agreement 100%).
#
# This is a defensible engineering choice rather than a defect. The exact
# focal Hessian is NEGATIVE for roughly 9% of instances at gamma = 2, and
# LightGBM requires a positive Hessian for a well-formed split gain; the
# modulated-weight form guarantees positivity. It is equivalent to gradient
# descent on a cross-entropy loss whose per-instance weights are recomputed
# from the current predictions at every iteration.
#
# But the manuscript claims focal loss as defined by Lin et al. (2017),
# implemented as a custom gradient/Hessian pair. Either the implementation
# or the claim has to change. Both forms are provided so the choice can be
# made explicitly and a sensitivity comparison reported.
#
#   "modulated"  (default) — the form used in R01 and R02. Reproduces every
#                            reported number. Positive Hessian guaranteed.
#   "exact"                — the true derivative of the focal loss, with the
#                            Hessian floored at HESSIAN_FLOOR for stability.
#
# Selected through config.FOCAL_GRADIENT; pipeline.fit_arm passes it through.

HESSIAN_FLOOR = 1e-6


def _focal_grad_hess_exact(y_true, y_pred, weight, gamma, alpha, eps):
    """Exact analytic derivative of the focal loss with respect to the raw
    margin. Matches a central-difference derivative to ~1e-9.

    L   = -alpha_t * (1 - pt)^gamma * log(pt)
    dL/dpt = alpha_t * (1 - pt)^(gamma-1) * (gamma*log(pt) - (1 - pt)/pt)
    dpt/dz = s * p * (1 - p),   s = +1 if y == 1 else -1
    """
    y_true = np.asarray(y_true, dtype=float)
    z = np.asarray(y_pred, dtype=float)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), eps, 1.0 - eps)
    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    s = np.where(y_true == 1, 1.0, -1.0)

    dL_dpt = alpha_t * (1.0 - pt) ** (gamma - 1.0) * (
        gamma * np.log(pt) - (1.0 - pt) / pt)
    grad = dL_dpt * s * p * (1.0 - p)

    # Second derivative by central difference on the analytic gradient: the
    # closed form is long and error-prone, and this is exact to ~1e-6.
    h = 1e-5

    def _g(zz):
        pp = np.clip(1.0 / (1.0 + np.exp(-zz)), eps, 1.0 - eps)
        ppt = np.where(y_true == 1, pp, 1.0 - pp)
        d = alpha_t * (1.0 - ppt) ** (gamma - 1.0) * (
            gamma * np.log(ppt) - (1.0 - ppt) / ppt)
        return d * s * pp * (1.0 - pp)

    hess = (_g(z + h) - _g(z - h)) / (2.0 * h)
    # LightGBM needs a positive Hessian; the exact focal Hessian is negative
    # for well-classified instances at gamma >= 1.
    hess = np.maximum(hess, HESSIAN_FLOOR)

    if weight is not None:
        w = np.asarray(weight, dtype=float)
        grad = grad * w
        hess = hess * w
    return grad, hess


def make_focal_variant(gamma=GAMMA, alpha=ALPHA, eps=EPSILON, form="modulated"):
    """Factory selecting the gradient form. See the note above."""
    if form not in ("modulated", "exact"):
        raise ValueError(f"unknown focal gradient form: {form!r}")
    if form == "modulated":
        return make_focal(gamma, alpha, eps)

    def focal_objective_exact(y_true, y_pred, weight):
        return _focal_grad_hess_exact(y_true, y_pred, weight, gamma, alpha, eps)

    def focal_eval_exact(y_true, y_pred):
        return "focal_loss", _focal_value(y_true, y_pred, gamma, alpha, eps), False

    focal_objective_exact.gamma = gamma
    focal_objective_exact.alpha = alpha
    focal_objective_exact.form = "exact"
    return focal_objective_exact, focal_eval_exact


def gradient_check(gamma=GAMMA, alpha=ALPHA, n=500, seed=0):
    """Compare both forms against a numerical derivative of the focal loss.
    Returns a dict; called by Notebook 6b so the comparison is committed."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    z = rng.normal(0, 2, n)
    h = 1e-6

    def L(zz):
        return np.array([_focal_value(y[i:i+1], zz[i:i+1], gamma, alpha, EPSILON)
                         for i in range(len(zz))])

    num = (L(z + h) - L(z - h)) / (2 * h)
    g_mod, h_mod = _focal_grad_hess(y, z, None, gamma, alpha, EPSILON)
    g_exa, h_exa = _focal_grad_hess_exact(y, z, None, gamma, alpha, EPSILON)
    return {
        "gamma": gamma, "alpha": alpha, "n": n,
        "modulated_max_abs_error": float(np.abs(g_mod - num).max()),
        "modulated_corr_with_true": float(np.corrcoef(g_mod, num)[0, 1]),
        "modulated_sign_agreement": float((np.sign(g_mod) == np.sign(num)).mean()),
        "exact_max_abs_error": float(np.abs(g_exa - num).max()),
        "modulated_hessian_min": float(h_mod.min()),
        "exact_hessian_pct_floored": float((h_exa <= HESSIAN_FLOOR * 1.000001).mean()),
    }
