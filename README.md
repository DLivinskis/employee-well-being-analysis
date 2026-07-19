# Employee Well-Being Analysis

A data product built on the Kaggle "Remote Work and Mental Health" dataset, exploring what drives employee satisfaction with remote work and predicting satisfaction for individual employees.

## Documentation

- [Running the App End-to-End](docs/running_the_app.md) — setup and run instructions, with `uv` or plain `pip` options
- [Task Description](docs/task_description.md) — the original assignment brief
- [AI-Suggested Approach](docs/ai_suggested_approach.md) — initial EDA plan, modeling strategy, and product-form ideas
- [Architecture and Design](docs/architecture_and_design.md) — the actual interface and system design being built
- [Execution Plan](docs/execution_plan.md) — repo structure and task breakdown for implementing the architecture
- [EDA Findings](docs/eda_findings.md) — data quality notes and feature-vs-target signal strength
- [Statistics Concepts](exploratory_analysis/statistics_concepts.md) — chi-square and ANOVA explained, as used in `descriptive_stats.py`
- [Multiclass AUC](training/multiclass_auc.md) — how one-vs-rest ROC/AUC is calculated for the 3-class target, as used in `evaluation.py`
- [Macro-F1](training/macro_f1.md) — how precision/recall/F1 are averaged across classes, with a worked example, as used in `evaluation.py`
- [Feature Importance Methods](training/feature_importance_methods.md) — standardized coefficients vs. permutation importance, how to interpret each, and what a negative value means for each
- [Metrics and Target Encoding Ideas](training/metrics_and_target_encoding_ideas.md) — not-yet-implemented ideas on ordinal target encoding and additional metrics like log loss
