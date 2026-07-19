# Statistics Concepts Used in Descriptive Stats

Background for the two significance tests `exploratory_analysis/descriptive_stats.py`'s `DescriptiveStats.feature_vs_target` runs — one per categorical feature, one per numeric feature — and how their outputs feed into `docs/eda_findings.md`.

## Chi-square test of independence (categorical features)

Used for every column in `CATEGORICAL_COLUMNS` (e.g. `Gender`, `Work_Location`, `Stress_Level`) against the target.

- **Question it answers:** are two categorical variables independent, or does knowing one tell you something about the other? Here: does an employee's category on this feature (e.g. `Work_Location = Remote`) shift the distribution of their satisfaction class, versus what you'd expect if the feature and satisfaction were unrelated?
- **How it works:** builds a crosstab of feature category × target class (the `counts` table `_categorical_vs_target` returns), computes what each cell *would* contain if the two variables were perfectly independent (based on the row/column totals alone), and measures how far the *observed* counts deviate from those *expected* counts. Bigger deviation → larger chi-square statistic → smaller p-value.
- **Assumptions worth knowing:** works best when expected cell counts aren't tiny (a common rule of thumb is ≥5 per cell); with 5,000 rows and a handful of categories per feature here, that's comfortably satisfied.
- **What it can't tell you:** *how strongly* the variables are related, or in which direction — only whether the association is stronger than chance would produce. It also treats every category independently, so it won't reveal "this one specific category is the driver" without also looking at `proportions`.

### Basic example

Suppose 80 employees split by `Work_Location` (`Remote` / `Onsite`) and a (simplified, 2-class) target `Satisfaction` (`Satisfied` / `Unsatisfied`):

| | Satisfied | Unsatisfied | Row total |
|---|---|---|---|
| Remote | 30 | 10 | 40 |
| Onsite | 10 | 30 | 40 |
| **Column total** | 40 | 40 | **80** |

If `Work_Location` and `Satisfaction` were independent, each cell's *expected* count would be `(row total × column total) / grand total` — e.g. the Remote/Satisfied cell would expect `40 × 40 / 80 = 20`, and by the same formula every cell here expects 20. The chi-square statistic sums `(observed − expected)² / expected` over all four cells:

```
(30-20)²/20 + (10-20)²/20 + (10-20)²/20 + (30-20)²/20
= 5 + 5 + 5 + 5 = 20.0
```

With 1 degree of freedom (`(rows−1) × (columns−1) = 1×1`), a chi-square statistic of 20.0 gives p ≈ 0.000008 — far below 0.05, so this toy table shows a genuine association (Remote employees skew heavily satisfied, Onsite skew heavily unsatisfied). Compare this to the real `Work_Location` result in `eda_findings.md` (p = 0.006, three target classes instead of two, and a much smaller observed/expected gap) — still significant, but nowhere near this toy example's strength.

## One-way ANOVA (numeric features)

Used for every column in `NUMERIC_COLUMNS` (e.g. `Age`, `Hours_Worked_Per_Week`, `Work_Life_Balance_Rating`) against the target.

- **Question it answers:** does this numeric feature's average value differ across the target's groups (`Satisfied` / `Neutral` / `Unsatisfied`), by more than you'd expect from random sampling variation alone?
- **How it works:** splits the numeric column into one sample per target class (the `groups` list built in `_numeric_vs_target`), then compares the variance *between* those group means to the variance *within* each group. If the groups' means are far apart relative to how spread out each group is internally, that's evidence the target actually separates the feature's values; if the between-group differences are small relative to the natural within-group noise, it looks like the groups could all be draws from the same underlying distribution.
- **Assumptions worth knowing:** classically assumes each group is roughly normally distributed with similar variances. With ~1,600+ rows per target class here, ANOVA's result is fairly robust to mild violations of those assumptions (large samples).
- **What it can't tell you:** *which* group(s) differ from which — a significant ANOVA says "at least one class differs," not "Satisfied is higher than Unsatisfied specifically." `group_means` is what you'd look at to see the direction, but distinguishing a real pairwise difference from noise would need a proper post-hoc test (e.g. Tukey's HSD), which isn't implemented here since the initial triage pass didn't call for it.

### Basic example

Suppose a numeric feature (e.g. `Work_Life_Balance_Rating`) split into three tiny target groups:

| Group | Values | Mean |
|---|---|---|
| Unsatisfied | 1, 2, 3 | 2 |
| Neutral | 4, 5, 6 | 5 |
| Satisfied | 7, 8, 9 | 8 |

The grand mean across all 9 values is 5. ANOVA compares two variances:

- **Between-group variance** — how far each group's mean is from the grand mean, weighted by group size: groups are centered at 2, 5, 8 vs. a grand mean of 5 — a large spread.
- **Within-group variance** — how spread out the values are *inside* each group: each group is just three consecutive integers — very little spread.

Because the groups' means are far apart relative to how tight each group is internally, the F-statistic comes out large (F ≈ 27 for this toy data, p ≈ 0.001) — strong evidence the groups have different true means. Contrast this with the real numeric features in `eda_findings.md` (all p > 0.1): there, each group's values overlap heavily and the group means sit close together relative to their spread, which is exactly what "no real relationship" looks like in this test.

## Chi-square vs. Pearson correlation

Easy to conflate since both get called "measuring association," but they answer different questions and apply to different data:

| | Chi-square test | Pearson correlation |
|---|---|---|
| **Applies to** | two categorical variables | two numeric (continuous) variables |
| **Output** | a test statistic → a p-value | a coefficient `r` between −1 and +1 |
| **What the output means** | how unlikely this data is *if the variables were independent* — a significance test | the strength **and direction** of a *linear* relationship — an effect-size measure |
| **Scales with sample size?** | yes — the same underlying association produces a bigger statistic (smaller p) with more rows, since more data makes you more confident the deviation from independence is real | no — `r` doesn't grow with more rows; more rows just make the estimate of `r` more precise |
| **Tells you direction?** | no — only that categories and target aren't independent, not which category pushes the target which way (you'd read `proportions` for that) | yes — sign of `r` tells you whether the variables move together (+) or oppositely (−) |

So they're not interchangeable: you can't compute a Pearson `r` on `Work_Location` (categories aren't numbers to correlate) or a meaningful chi-square on `Age` (continuous values would mostly have exactly one row per cell in a crosstab).

**Where point-biserial correlation fits in:** it's Pearson correlation applied when one variable is numeric and the other is binary (exactly 2 categories) — e.g. correlating `Hours_Worked_Per_Week` with a `Satisfied` vs. `not-Satisfied` split gives both a direction and a strength, something ANOVA alone doesn't. It doesn't apply directly here because the real target has *three* classes, not two — ANOVA (or point-biserial computed pairwise per class) is the generalization `_numeric_vs_target` uses instead. This is why `descriptive_stats.py` uses ANOVA rather than a plain correlation coefficient for the numeric features: ANOVA handles the 3-class target directly, at the cost of not returning a signed "direction" the way `r` would for a 2-class target.

## Why p-values from both tests are compared side by side

Chi-square and ANOVA test different kinds of relationships (categorical association vs. group-mean difference) but both reduce to the same kind of output — a p-value under "no real relationship" as the null hypothesis — which is precisely why `feature_vs_target` can rank *all* features, categorical and numeric alike, on a common footing in the summary table in `docs/eda_findings.md`. The multiple-comparisons caveat in that doc (17 independent tests, so expect ~1 false positive at p<0.05 by chance) applies identically regardless of which of the two tests produced the number.
