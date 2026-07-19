# Ideas: Alternative Metrics and Target Encoding

Notes on two related, not-yet-implemented ideas for `training/`, prompted by questions about how `Satisfaction_with_Remote_Work` and its evaluation metrics currently work. Nothing here is implemented — this is a list of options and tradeoffs to revisit, not a plan committed to.

## 1. How the target is currently encoded

`Satisfaction_with_Remote_Work` stays as raw strings (`"Satisfied"`, `"Neutral"`, `"Unsatisfied"`) all the way into `LogisticRegression`/`RandomForestClassifier` (see `training/train_logistic.py`, `training/train_tree.py`). scikit-learn maps those to integers internally, but by sorting the strings **alphabetically** (`classes_` comes out `['Neutral', 'Satisfied', 'Unsatisfied']`) — not `{-1, 0, 1}` reflecting the actual satisfaction ordering. The models are trained as plain **nominal** multiclass classification: they get no signal that `Unsatisfied` and `Satisfied` are further apart than `Unsatisfied` and `Neutral`, and the confusion matrix / ROC-AUC / macro-F1 metrics currently used don't assume or need any ordering either — that's exactly what they're built for.

### Option: treat it as ordinal instead

`Satisfied` > `Neutral` > `Unsatisfied` is a genuine ordering, and the current setup ignores it. If that ordering turned out to matter (e.g. because a bigger dataset revealed a model that's directionally right even when it misses the exact class), options to actually exploit it:

- **Encode as `{-1, 0, 1}` and use regression**, then bucket the continuous prediction back into three classes for reporting. Simple, but loses calibrated class probabilities (what `/predict` currently returns) unless a separate calibration step is added.
- **Ordinal logistic regression / proportional-odds model** (e.g. `mord` or a manually implemented cumulative-link model) — the "correct" statistical tool for an ordered target, since it models P(satisfaction ≤ k) directly rather than treating each class as independent.
- **Keep plain multiclass, but add an ordinal-aware metric** (see Mean Absolute Error below) without changing the models themselves — the cheapest option, since it doesn't touch `train_logistic.py`/`train_tree.py` at all, just `evaluation.py`.

### Cost of switching — two very different scopes

"Change the encoding to `-1/0/1`" can mean two things with very different costs:

- **Just relabeling the strings, keeping the current classifiers (trivial, ~10 minutes, no real benefit).** `LogisticRegression`/`RandomForestClassifier` are plain classifiers — they treat every class as an equally-distinct category no matter what the label is. Internally they'd map `-1/0/1` to indices `0/1/2` exactly the way they currently map the three strings to `0/1/2` alphabetically; the numeric distance between `-1` and `1` is never used by a classifier's decision logic. Swapping labels alone would just be cosmetic renaming in `run_training.py`'s data loading and the label lists in `evaluation.py` — it would change nothing about what the model learns.
- **Actually exploiting the ordering (moderate — touches every layer).** To get real value from `-1/0/1`, you'd need one of the genuinely ordinal approaches above (regression + bucketing, or a proper ordinal/proportional-odds model), and that ripples outward well beyond `training/`:
  - `training/train_logistic.py`/`train_tree.py` — a different estimator type, likely without `predict_proba` in the same sense.
  - `training/helpers/evaluation.py` — confusion matrix and macro-F1 still work on bucketed predictions, but ROC/AUC doesn't apply the same way to a regression output; you'd add MAE-on-ordinal-encoding (below) instead.
  - `api/endpoint_helpers/schemas.py` and `inference.py` — `PredictResponse.class_probabilities` assumes calibrated per-class probabilities from `predict_proba`; a regression-based approach needs its own translation into something probability-shaped, or the response contract changes.
  - `frontend/pages/2_Logistic_Model_Features.py`/`3_Tree_Model_Features.py` — the per-class coefficient/importance and ROC-AUC rendering would need rework, since there's no longer a natural per-class probability output the same way.

  Realistically a few hours of work across three layers (`training/`, `api/`, `frontend/`), not a one-line change.

### Why this probably isn't the first thing to fix

Given `docs/eda_findings.md`'s finding that almost no feature shows real signal against the target, switching modeling paradigms is unlikely to move the needle much — the near-chance macro-F1 is much more plausibly explained by the dataset having little signal at all, not by discarding ordinal information. Worth revisiting only after (or if) a feature is found that shows a real, ordered relationship worth modeling that way.

## 2. Metrics beyond confusion matrix / ROC-AUC / macro-F1

The three metrics `training/helpers/evaluation.py` currently computes are all standard, threshold-based-or-ranking multiclass metrics that don't assume ordering. A few alternatives, and what each would actually add:

- **Log loss (cross-entropy)** — scores the predicted *probabilities* directly against the true class, penalizing confident-but-wrong predictions harder than ROC/AUC does (which only cares about rank-ordering, not calibration). This is the one alternative that adds genuinely new information here, since `/predict` shows raw class probabilities to the end user (`docs/architecture_and_design.md`'s Predict page) — if those probabilities are poorly calibrated, ROC/AUC and macro-F1 wouldn't catch it, but log loss would. Cheap to add: `sklearn.metrics.log_loss(y_true, y_proba)`, one line in `ModelEvaluator.evaluate`.
- **Precision-Recall AUC** — same one-vs-rest idea as ROC/AUC (see `training/multiclass_auc.md`), but plots precision vs. recall instead. More informative than ROC/AUC specifically when a class is rare, which isn't the case here — the three classes are close to evenly split (per `docs/eda_findings.md`), so PR-AUC would track ROC/AUC closely and add little.
- **Mean Absolute Error on an ordinal encoding** — if the target were mapped to `{-1, 0, 1}` (or `{0, 1, 2}`) as in the ordinal option above, MAE between true and predicted encoded values would reward "off by one class" less harshly than "off by two classes" (e.g. predicting `Neutral` when the truth is `Satisfied` is a smaller error than predicting `Unsatisfied`). None of the three current metrics express this — a confusion matrix shows *which* cells are wrong but doesn't score "how wrong." This only makes sense paired with the ordinal-encoding option above; it's meaningless against the current nominal encoding.
- **Cohen's kappa / balanced accuracy** — single-number summaries similar in spirit to macro-F1 (agreement corrected for chance). Redundant with macro-F1 here rather than adding new information, given classes are already balanced.

### If one were added first

Log loss is the strongest candidate: it's cheap (one more line, no new dependency), doesn't require changing how the target is encoded or how the models are trained, and answers a question the current metrics genuinely can't — whether the probabilities the app actually displays to users are trustworthy.
