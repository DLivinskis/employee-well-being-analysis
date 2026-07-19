# Calculating Macro-F1

Background for `training/helpers/evaluation.py`'s `ModelEvaluator._macro_f1`, which scores both models on the 3-class `Satisfaction_with_Remote_Work` target.

## Precision, recall, and F1 for one class

For a single class (say `Satisfied`), treated one-vs-rest (same framing as `training/multiclass_auc.md`):

- **Precision** — of the rows the model *predicted* as `Satisfied`, what fraction actually were `Satisfied`.
- **Recall** — of the rows that *actually are* `Satisfied`, what fraction did the model catch.
- **F1** — the harmonic mean of the two: `F1 = 2 · (precision · recall) / (precision + recall)`. The harmonic mean (rather than a plain average) means F1 stays low if *either* precision or recall is low — a model can't hide a bad recall behind a good precision, or vice versa.

## Macro-F1: averaging across classes

Compute F1 independently for each of the three classes, then take the plain, unweighted mean of the three — that's macro-F1. Every class counts equally regardless of how many rows it has (as opposed to a *weighted* average, which would weight each class's F1 by its row count).

## A basic worked example

9 rows, 3 true `Satisfied`, 3 true `Neutral`, 3 true `Unsatisfied`. Confusion matrix (rows = true, columns = predicted, order `[Satisfied, Neutral, Unsatisfied]`), verified against `sklearn.metrics.f1_score`:

|              | pred: Satisfied | pred: Neutral | pred: Unsatisfied |
|--------------|:---:|:---:|:---:|
| **true: Satisfied**   | 2 | 1 | 0 |
| **true: Neutral**     | 0 | 3 | 0 |
| **true: Unsatisfied** | 2 | 0 | 1 |

Working through each class:

- **Satisfied**: predicted 4 times total (2 correct + 2 rows that were actually `Unsatisfied`), so precision = 2/4 = **0.5**. Of 3 true `Satisfied` rows, 2 were caught, so recall = 2/3 = **0.667**. F1 = 2·(0.5·0.667)/(0.5+0.667) = **0.571**.
- **Neutral**: predicted 4 times (3 correct + 1 row that was actually `Satisfied`), precision = 3/4 = **0.75**. All 3 true `Neutral` rows were caught, recall = 3/3 = **1.0**. F1 = 2·(0.75·1.0)/(0.75+1.0) = **0.857**.
- **Unsatisfied**: predicted only once, and correctly, so precision = 1/1 = **1.0**. But only 1 of the 3 true `Unsatisfied` rows was caught (the other 2 were predicted as `Satisfied`), recall = 1/3 = **0.333**. F1 = 2·(1.0·0.333)/(1.0+0.333) = **0.5**.

**Macro-F1** = (0.571 + 0.857 + 0.5) / 3 = **0.643**.

## Why this differs from plain accuracy

Plain accuracy on this same example is 6/9 = **0.667** (6 rows predicted correctly out of 9) — higher than macro-F1's 0.643. The gap is `Unsatisfied`'s poor recall (0.333): two-thirds of the true `Unsatisfied` rows were missed (both misclassified as `Satisfied`), which accuracy mostly absorbs into the overall count, but macro-F1 penalizes directly by weighting that class's poor F1 equally with the other two. This is exactly the property that makes macro-F1 preferred over accuracy here — it surfaces a model that's lopsided across classes rather than letting good performance on one class mask poor performance on another, per `docs/architecture_and_design.md`'s reasoning for choosing it.
