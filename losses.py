
# losses.py
#
# Single source of truth for the focal-loss objective used by both
# Notebook 6b (training E-LightGBM) and Notebook 7 (SHAP explanation).
# Save this file to PROJECT_DIR (/content/drive/MyDrive/Ghana_Dropout_Project/losses.py)
# so both notebooks can `sys.path.append(PROJECT_DIR)` and `from losses import ...`.
#
# Having exactly one copy of this code is deliberate: the argument-order
# bug found earlier happened precisely because this logic was copy-pasted
# across files and edited inconsistently. Import this module everywhere
# rather than redefining these functions locally.

import numpy as np

# Focusing exponent (downweights easy, well-classified examples) and
# class-weighting scalar (upweights the minority dropout class),
# matching the parameterisation written into the Methods section.
GAMMA = 2.0
ALPHA = 0.75
EPSILON = 1e-6


def focal_loss_lgb(y_true, y_pred):
    """Custom gradient/Hessian pair implementing binary focal loss
    (Lin et al., 2017), supplied to LightGBM via the `objective`
    parameter in place of the default binary cross-entropy loss.

    IMPORTANT: LightGBM's sklearn API (LGBMClassifier) calls a custom
    objective positionally as objective(y_true, y_pred), where both
    arguments are plain numpy arrays - NOT a Dataset object. Do not
    call .get_label() on either argument here; that convention only
    applies to the native lgb.train(params, dtrain, ...) API, not the
    sklearn wrapper used throughout this project.

    The parameter order below MUST stay (y_true, y_pred). Reversing it
    silently swaps the true labels and the raw prediction scores and
    will cause the model to collapse to predicting the majority class
    (observable as AUC-ROC = 0.500 and zero recall on the minority
    class).
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, EPSILON, 1.0 - EPSILON)

    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, ALPHA, 1.0 - ALPHA)

    grad = alpha_t * (1.0 - pt) ** GAMMA * (p - y_true)
    hess = alpha_t * (1.0 - pt) ** GAMMA * p * (1.0 - p)

    return grad, hess


def focal_loss_eval(y_true, y_pred):
    """Focal loss value (not just grad/hess) for monitoring during
    training. Returns (eval_name, eval_result, is_higher_better).
    Same argument-order requirement as focal_loss_lgb above - do not
    call .get_label() on either argument.
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, EPSILON, 1.0 - EPSILON)

    pt = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, ALPHA, 1.0 - ALPHA)

    loss = -alpha_t * (1.0 - pt) ** GAMMA * np.log(pt)

    return "focal_loss", float(np.mean(loss)), False


def predict_proba_focal(model, X_eval):
    """LightGBM's built-in .predict_proba() does NOT work correctly
    under a custom objective - it returns raw margin scores (logits),
    not calibrated probabilities. Always use this function to get
    actual probabilities from a model trained with focal_loss_lgb.
    """
    raw_scores = model.predict(X_eval, raw_score=True)
    return 1.0 / (1.0 + np.exp(-raw_scores))
