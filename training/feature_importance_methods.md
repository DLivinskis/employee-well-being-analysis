# Standardized Coefficients and Permutation Importance

Background for the two feature-importance techniques `training/train_logistic.py` and `training/train_tree.py` use, how to read their numbers, and — the part that trips people up — what a **negative** value means for each.

## Standardized coefficients (logistic model)

A plain logistic regression coefficient tells you how much the model's log-odds change per one-unit increase in a feature — but "one unit" means something different for every feature (one unit of `Age` vs. one unit of a 0/1 one-hot column vs. one unit of `Hours_Worked_Per_Week`), so raw coefficients aren't comparable across features. `train_logistic.py`'s pipeline applies `StandardScaler` right before `LogisticRegression` (the `scaling` step in `_build_pipeline`), which rescales every feature to mean 0, standard deviation 1. With that done, "one unit" uniformly means "one standard deviation" for every feature, so the resulting coefficients genuinely are comparable in magnitude — that's why `feature_importance` ranks them by `abs(coefficient)`.

Coefficients are computed **per class** (the model has three, one per satisfaction level), since a feature can push toward one class while pushing away from another.

### How to interpret a coefficient

For a given class (e.g. `Satisfied`), a coefficient on a feature says: holding every other feature fixed, how does a one-standard-deviation increase in this feature change the model's confidence in *this specific class*, relative to the others.

- **Positive** → higher values of this feature push the model *toward* predicting this class.
- **Negative** → higher values of this feature push the model *away from* this class (toward one of the other two).
- **Magnitude near zero** → the model isn't using this feature to distinguish this class from the others, regardless of sign.

### Real examples from this project

From the `Satisfied` class's coefficients in `data/models/metadata.json`:

- `Company_Support_for_Remote_Work`: **+0.050** — more company support for remote work standard-deviations above average nudges the model slightly toward predicting `Satisfied`.
- `Work_Location_Remote`: **−0.055** — being in the `Remote` category (vs. the other work-location categories) nudges the model slightly *away* from predicting `Satisfied` — i.e. slightly toward `Neutral`/`Unsatisfied` instead.

### What a negative coefficient means here specifically — and what to do about it

A negative coefficient is not a problem to fix — it's a legitimate, informative result: it means this feature and this class move in opposite directions. There's nothing to "do" about a negative sign by itself. What actually matters:

- **Look at the magnitude, not just the sign.** Both examples above are close to zero (±0.05) on a standardized scale — meaningfully small effects, consistent with `docs/eda_findings.md`'s finding that this dataset has almost no real signal. A large negative coefficient (e.g. −0.5 or beyond) would be worth a closer look; a small one near zero, positive or negative, isn't.
- **Check the same feature's sign across all three classes.** Since the three classes must together account for 100% of the probability mass, a feature that's negative for `Satisfied` is often positive for one of the other two — that's expected structure, not a contradiction to resolve.
- **Don't over-read a single small coefficient as a "finding."** With 17 features and 3 classes, some coefficients will land on the negative side of zero by chance alone, the same multiple-comparisons caution as in `exploratory_analysis/statistics_concepts.md`.

## Permutation importance (tree model)

A model-agnostic technique that doesn't inspect the model's internals — it empirically measures "how much worse does the model get if this one feature's information is destroyed?" `train_tree.py`'s `feature_importance` takes a held-out split (validation data, not training data — see the docstring's note on why), shuffles one raw feature column at a time (breaking its real relationship to the target while preserving its distribution), re-scores the model (`scoring="f1_macro"`), and records how much the score drops. This is repeated `n_repeats=10` times per feature and averaged, for stability.

This is used instead of the tree model's built-in impurity-based importance because impurity-based importance is biased toward high-cardinality features (e.g. `Job_Role` with 7 categories looks artificially more important than a binary column purely because it has more places to split on) — permutation importance doesn't have that bias, since it only measures the actual effect on predictive performance.

### How to interpret a permutation importance value

- **Larger positive value** → shuffling this feature hurt the model more, meaning the model was relying on it more.
- **Value near zero** → shuffling this feature barely changed the score — the model wasn't using it much.
- Unlike standardized coefficients, there's **no direction/sign to read** — permutation importance only tells you *how much* a feature matters, never *which way* it pushes the prediction.

### Real examples from this project

From `data/models/metadata.json`'s tree-model importances:

- `Work_Location`: **+0.024** (the largest here) — shuffling it costs the most macro-F1 of any feature, i.e. the model relies on it the most (still a small effect in absolute terms).
- `Hours_Worked_Per_Week`: **−0.0062** — shuffling this feature *improved* the score slightly.

### What a negative permutation importance means here specifically — and what to do about it

A negative value does **not** mean "this feature actively hurts the model" in a way worth removing it for. Since permutation importance is a difference of two noisy score estimates (score-with-real-data minus score-with-shuffled-data), a feature that has truly no real relationship to the target will scatter around zero across repeated shuffles — sometimes landing slightly positive, sometimes slightly negative, purely from sampling noise in which rows end up in which shuffle. A negative number here is simply the noisy side of "this feature has ~zero importance," not a meaningful finding in its own right.

What to actually do:

- **Treat small negative values the same as small positive values near zero** — both mean "not important," full stop. Don't rank a slightly-negative feature as "actively harmful" and a slightly-positive one as "helpful" when both are within noise of zero.
- **Don't drop features solely because their importance is negative.** If a feature's importance were meaningfully and consistently negative across re-runs with different seeds, that would be worth investigating (it could hint at overfitting-driven noise the feature is contributing) — but a single small negative value from one run isn't strong enough evidence to act on.
- **Compare against the largest importance in the same run for scale.** Here, the biggest importance is `Work_Location` at +0.024 — `Hours_Worked_Per_Week`'s −0.0062 is small relative to that, reinforcing that it's noise-level rather than a real negative effect.
