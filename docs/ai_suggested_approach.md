# Approach

## 1. What the data actually looks like

The raw file (`raw_data/Impact_of_Remote_Work_on_Mental_Health.csv`) has 5,000 rows and 20 columns: an ID, demographics (`Age`, `Gender`, `Region`), job context (`Job_Role`, `Industry`, `Years_of_Experience`, `Work_Location`), workload (`Hours_Worked_Per_Week`, `Number_of_Virtual_Meetings`), wellbeing signals (`Work_Life_Balance_Rating`, `Stress_Level`, `Mental_Health_Condition`, `Access_to_Mental_Health_Resources`, `Social_Isolation_Rating`, `Physical_Activity`, `Sleep_Quality`), and outcome-adjacent fields (`Productivity_Change`, `Company_Support_for_Remote_Work`).

The target for both brief questions is `Satisfaction_with_Remote_Work`: `Satisfied` / `Neutral` / `Unsatisfied`, split almost exactly evenly (~1,650–1,680 rows each).

**First judgment call:** before investing in feature engineering or modeling, check whether any feature actually correlates with the target. A quick pass over every categorical column's value counts shows near-uniform distributions across categories (e.g. `Stress_Level`, `Work_Location`, `Sleep_Quality` are all roughly evenly split), which is a common signature of a synthetically generated dataset with little embedded signal. If EDA confirms that no feature moves the target more than noise would, that's a legitimate finding to report rather than something to paper over with a more complex model — the brief asks what *actually* drives satisfaction, and "not much, in this dataset" is a valid, honest answer. This gets checked in step 2 before committing to a modeling strategy.

## 2. EDA plan

- Univariate: distribution of the target and of each candidate feature.
- Bivariate: target rate (or mean ordinal score) by category for every categorical feature; correlation/point-biserial for numeric features (`Age`, `Years_of_Experience`, `Hours_Worked_Per_Week`, `Number_of_Virtual_Meetings`, `Work_Life_Balance_Rating`, `Social_Isolation_Rating`).
- Chi-square / ANOVA-style significance checks per feature against the target, purely as a triage signal for which features are worth carrying into modeling — not as a substitute for the model's own feature importance.
- Sanity-check `Employee_ID` is a pure identifier (drop it) and check for missing values / inconsistent categories.

This step directly answers the "what should count as a meaningful factor" question the brief asks about, and determines whether question 1 gets answered with "these N features matter" or "the data doesn't support strong claims here, here's what's suggestive."

## 3. Feature engineering

Kept intentionally light, since the dataset is already tidy tabular data with no timestamps, text, or nested structure to derive from:

- One-hot / ordinal encode categoricals (ordinal for genuinely ordered ones like `Stress_Level`, `Sleep_Quality`, `Company_Support_for_Remote_Work`; one-hot for nominal ones like `Job_Role`, `Industry`, `Region`).
- Leave numeric ratings as-is; consider binning only if EDA shows a clearly non-linear relationship a linear model would miss.
- No feature is added without a specific reason found in EDA — the risk on a dataset this size and this flat is overfitting to noise with engineered combinations that "sound" plausible but aren't supported by the data.

## 4. Modeling approach

The target is a 3-class categorical outcome, so this is a multiclass classification problem, not regression.

- **Baseline:** multinomial logistic regression — fast, interpretable coefficients, gives an immediate read on whether there's linear signal at all.
- **Comparison model:** a tree ensemble (gradient boosting or random forest) to capture any non-linear or interaction effects the linear baseline misses, and to cross-check whether it meaningfully beats the baseline (if it doesn't, that itself supports the "weak signal" finding from EDA).
- **Validation:** stratified train/test split (or stratified k-fold given only 5,000 rows), since the classes are balanced but a hold-out still needs to preserve that balance for fair comparison.
- **Metric:** macro-F1 (treats all three satisfaction classes equally) alongside a confusion matrix, since accuracy alone is uninformative on a balanced 3-class problem and the brief cares about getting each satisfaction level right, not just the majority class.

## 5. Answering "what drives satisfaction" (question 1)

- Logistic regression coefficients (with standardized inputs, so magnitudes are comparable) for a first, interpretable pass.
- Permutation feature importance and/or SHAP values on the tree model as the primary answer, since they account for interactions and non-linearities the logistic model can't.
- Cross-check the two: features that show up as important in both are the trustworthy answer to question 1; anything that only shows up in one model gets flagged as tentative rather than presented as a firm conclusion.

## 6. Answering "predict satisfaction for a given employee" (question 2)

- Expose the trained classifier through a single `predict(employee_attributes) -> {label, class_probabilities}` function, so the *shape* of the prediction (probabilities per class, not just a hard label) is preserved regardless of what wraps it later.
- Keep the preprocessing (encoding, scaling) inside the same object as the model, so a caller only ever passes raw attribute values — not something they've had to remember to pre-encode correctly.

## 7. Product form

The brief leaves format open and says the thinking behind the choice matters more than the format. Given the scope (EDA insights + a single-employee predictor, no streaming data, no need for auth or multi-user state), a **notebook-driven report backed by a small reusable Python module** is proposed over a full web app or API:

- The reusable module (analysis + model classes) is what actually needs to be correct and testable — that's where the engineering judgment lives.
- A lightweight dashboard (e.g. Streamlit) on top of that module is worth adding *if* the reusable module is solid first, since it's a thin presentation layer over already-validated logic rather than the main deliverable.
- An agentic/conversational layer answering ad-hoc questions about the data is an interesting stretch goal, but only after the core analysis and prediction logic are solid — an agent wrapping shaky analysis just launders bad conclusions with a friendlier interface.

## 8. Stack

- `pandas` for data handling, `scikit-learn` for modeling/encoding/metrics, `shap` for interaction-aware feature importance, `matplotlib`/`seaborn` for EDA plots.
- Project structure: `raw_data/` (already set up with `fetch_kaggle_dataset.py`), `exploratory_analysis/` (the EDA notebook), and a to-be-added `src/` module for the reusable analysis + model classes once EDA findings settle what's worth encoding into them.
