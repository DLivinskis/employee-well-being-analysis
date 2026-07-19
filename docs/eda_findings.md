# EDA Findings

Computed via `exploratory_analysis/descriptive_stats.py`. This is the written summary [ai_suggested_approach.md](ai_suggested_approach.md) flagged as a prerequisite before committing to a modeling strategy.

## 1. Data quality

- **No true missing values, but `read_csv` needs `keep_default_na=False, na_values=[]`.** `Mental_Health_Condition` and `Physical_Activity` use the literal string `"None"` as a real category (e.g. "no mental health condition"), and pandas' default NA parsing silently turns those into `NaN` — pandas reports 1,196 and 1,629 "missing" values respectively that aren't actually missing. Every read of this CSV (training, seeding, EDA) must disable default NA parsing to avoid quietly dropping a real category.
- `Employee_ID` is a unique key for all 5,000 rows — safe to use as a lookup key and to drop as a model feature.
- Target `Satisfaction_with_Remote_Work` is balanced: `Unsatisfied` 1,677, `Satisfied` 1,675, `Neutral` 1,648.

## 2. Feature-vs-target signal

Chi-square (categorical features) and one-way ANOVA (numeric features) against the target, from `DescriptiveStats.feature_vs_target()`:

| Feature | Test | p-value |
|---|---|---|
| `Work_Location` | chi-square | **0.006** |
| `Years_of_Experience` | anova | 0.186 |
| `Social_Isolation_Rating` | anova | 0.192 |
| `Gender` | chi-square | 0.230 |
| `Company_Support_for_Remote_Work` | anova | 0.229 |
| `Stress_Level` | chi-square | 0.289 |
| `Number_of_Virtual_Meetings` | anova | 0.315 |
| `Productivity_Change` | chi-square | 0.350 |
| `Region` | chi-square | 0.423 |
| `Access_to_Mental_Health_Resources` | chi-square | 0.470 |
| `Hours_Worked_Per_Week` | anova | 0.749 |
| `Mental_Health_Condition` | chi-square | 0.729 |
| `Sleep_Quality` | chi-square | 0.733 |
| `Physical_Activity` | chi-square | 0.784 |
| `Job_Role` | chi-square | 0.881 |
| `Age` | anova | 0.985 |
| `Industry` | chi-square | 0.920 |

**Conclusion: this confirms the hypothesis raised in `ai_suggested_approach.md`.** Only `Work_Location` clears the conventional p < 0.05 bar, and even that one wouldn't survive a multiple-comparisons correction across 17 tests (Bonferroni threshold ≈ 0.003). Every other feature is consistent with pure noise against this target. This matches the earlier observation that categorical distributions look close to uniform — a signature of synthetically generated data with no strong embedded relationships.

## 3. Implications for modeling (Phase 2)

- **No features are dropped from the model based on this alone** — `Work_Location`'s borderline signal and the trained models' own feature importances (logistic coefficients, tree SHAP/permutation importance) are the real answer to "what drives satisfaction," per [architecture_and_design.md](architecture_and_design.md). Statistical univariate tests here are a triage signal, not the final word — they don't capture interactions a tree model might still find.
- **Honest reporting matters more than model complexity here.** If both trained models also show weak, inconsistent importances and near-chance accuracy, that itself is the answer to the brief's question 1 ("what are the primary factors") — not a signal to keep engineering features or tuning hyperparameters until something looks stronger.
- Preprocessing (`training/preprocessing.py`) still encodes every feature, since it's the trained models — not this univariate pass — that make the final call on importance.
