# Calculating AUC for a Multiclass Model

Background for `training/helpers/evaluation.py`'s `ModelEvaluator._roc_auc_per_class`, which computes ROC/AUC for the 3-class `Satisfaction_with_Remote_Work` target (`Satisfied` / `Neutral` / `Unsatisfied`).

## Why plain ROC/AUC doesn't apply directly

ROC curves and AUC are inherently a **binary** concept: at every possible decision threshold, plot the true positive rate against the false positive rate for "is this the positive class or not." That only makes sense when there are exactly two classes to call "positive" and "negative."

With three classes, there's no single natural "positive" class — the model instead outputs three probabilities per row (one per class, summing to 1, from `predict_proba`), and "is the model good at distinguishing `Satisfied` from everything else" is a genuinely different question from "is it good at distinguishing `Unsatisfied` from everything else." One curve and one number can't capture that.

## The approach used here: one-vs-rest (OvR)

For each class, temporarily treat the problem as binary — "is this row that class, or is it one of the other classes" — and compute an ordinary binary ROC/AUC for that reframing. Repeating this once per class gives one ROC curve and one AUC per class, which is exactly what `_roc_auc_per_class` returns.

Concretely, for the `Satisfied` class:

1. **Binarize the true labels**: every row where the true label is `Satisfied` becomes `1`; every row where it's `Neutral` or `Unsatisfied` becomes `0`. This is what `label_binarize` does in the code — it produces one 0/1 column per class.
2. **Take that class's probability column**: `predict_proba`'s `Satisfied` column, i.e. how confident the model is that each row is `Satisfied` (regardless of which of the other two classes it might otherwise be).
3. **Compute an ordinary binary ROC curve/AUC** from those two arrays (step 1's binarized labels, step 2's probabilities), the same way you would for any binary classifier.

Step 3 is why the code can just call `sklearn.metrics.roc_curve`/`roc_auc_score` directly, once per class — from that point on it genuinely is a binary problem.

## Why `proba_columns` (not `class_labels`) drives the loop

`predict_proba` returns columns in whatever order the fitted model's own `classes_` attribute uses — scikit-learn sorts labels alphabetically for this, but relying on that implicitly is fragile. `_roc_auc_per_class` takes `pipeline.classes_` in explicitly (as `proba_columns`) and binarizes using that exact same order, so column `i` of `y_proba` is always paired with the correct label at index `i`. Assuming some other ordering (e.g. the `class_labels` list `ModelEvaluator` was constructed with) would silently compute the wrong class's AUC if the two orderings ever diverged.

## A basic worked example

Suppose 6 rows, true labels `[Satisfied, Satisfied, Neutral, Neutral, Unsatisfied, Unsatisfied]`, and the model's predicted probability of `Satisfied` for each row: `[0.9, 0.6, 0.3, 0.7, 0.2, 0.1]`.

**Step 1 — binarize for the `Satisfied` class**: `[1, 1, 0, 0, 0, 0]`.

**Step 2 — the `Satisfied` probability column**: `[0.9, 0.6, 0.3, 0.7, 0.2, 0.1]`.

**Step 3 — sweep thresholds** and compute true/false positive rates. At threshold 0.65 (say "predict positive if probability ≥ 0.65"): rows 1 and 4 would be called positive (probabilities 0.9, 0.7); only row 1 is truly `Satisfied`. That's 1 true positive out of 2 actual positives (TPR = 0.5) and 1 false positive out of 4 actual negatives (FPR = 0.25) — one point on the ROC curve. Sweeping every threshold this way (verified against `sklearn.metrics.roc_curve` on exactly these six numbers) traces the full curve, and `sklearn.metrics.roc_auc_score` on the same data gives **AUC = 0.875**. In this toy example the one negative row with a high `Satisfied` probability (row 4, at 0.7) is what keeps AUC below a perfect 1.0 — it's a case where the model was fooled.

This exact procedure — repeated independently for `Neutral` and `Unsatisfied` — is what produces the three `{fpr, tpr, auc}` entries `_roc_auc_per_class` returns, and what the Streamlit model pages (`frontend/pages/2_Logistic_Model_Features.py`, `3_Tree_Model_Features.py`) plot as three separate ROC curves via `frontend/helpers/model_analysis.py`'s `render_roc_curves`.

## A note on averaging (not currently used)

One-vs-rest naturally produces per-class AUCs. If a single summary number were ever needed instead (analogous to how `macro_f1` summarizes per-class F1), the standard options are:

- **Macro-average**: the plain mean of the per-class AUCs, treating every class equally regardless of how many rows it has.
- **Weighted average**: the mean weighted by each class's row count.

Neither is currently exposed by the API/frontend — the per-class breakdown is more informative here, and with `Satisfied`/`Neutral`/`Unsatisfied` already close to evenly sized (per `docs/eda_findings.md`), macro and weighted averages would land very close to each other anyway.
