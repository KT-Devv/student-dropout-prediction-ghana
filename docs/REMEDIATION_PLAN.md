# Group 5 — Testbed Remediation Plan (R01 → R02)

Worked from `GROUP_5_TESTBED_FEEDBACK.docx`, the submitted Method and Results
sections, and all thirteen notebooks plus `losses.py`.

Four gates failed and 22 items scored RED. The examiner's own root-cause
analysis is right: **two causes carry more than half the list.** Fix the arm
definitions and re-run everything through one in-fold pipeline and about a
dozen items close at once. That is what `Notebook_10_Remediation.ipynb` does.

---

## 0. Three decisions only your group can make

Everything downstream branches on these. Make them first, in writing, with
your supervisor.

**D1 — What is the unit of the claim?**
Four schools, one holding half the sample, dropout rates spanning 3.3% to
20.7%, and `school_code` sitting 10th in the SHAP ranking. Two defensible
answers:

- *Pupil-level within these four schools.* Drop `school_code` from the
  feature matrix and say so. Simpler, honest, and the notebook default
  (`SCHOOL_HANDLING = "drop"`).
- *Any school-level reading.* Then you need `GroupKFold`/leave-one-school-out
  and must report that the effective cluster count is four.

Recommendation: take the first as primary, and report the leave-one-school-out
result (Section 10 of the notebook) as a bounded secondary analysis. Four
clusters cannot support a grouped primary design, and **saying that is itself
a finding worth a sentence.**

**D2 — Which contrast is the headline?**
The examiner is right that your own Table 4 already contains the clean
contrast. But the notebook goes one better: the matched pair
`E_all_ce_noW` vs `G_all_focal_noW` is identical in feature matrix *and*
weighting and differs only in the `objective` argument. Promote that.
Retire −0.013 explicitly as a three-change contrast; retire −0.004 as a
two-change contrast (it also drops the imbalance correction).

**D3 — Which theory?**
Pathway B was declared, so the theory must be chosen to fit the observed
attribution. Neither EST nor SDT does. The examiner's candidates — Finn's
participation–identification model as primary, household human-capital
investment under constraint as supporting — fit the three-family pattern your
SHAP output actually shows. Read the primary sources; do not cite from the
examiner's summary or from this document. Decide before you touch the
Introduction, because this one is not fixable after submission.

---

## 1. Work packages, in dependency order

Total ≈ 8 working days. Compute is not the constraint; every model fits in
under a second.

| # | Package | Closes | Blocked by | Days |
|---|---|---|---|---|
| **WP1** | Run `Notebook_10` Sections 1–4. Fix column names, read the cascade, missingness, outlier and cluster tables. | Q2, Q11, GATE-1(iv) | D1 | 0.5 |
| **WP2** | Sections 5–8: matched arms, full 2×2×2 grid, ten seeds, two-level variance, bootstrap CI. | GATE-1(i,ii), GATE-2, Q6, Q20, Q23a/c | WP1, D2 | 1.5 |
| **WP3** | Sections 9–11: γ×α grid, leave-one-school-out, instance-level stratification. | Q8, Q17, Q23h, Q24 | WP2 | 1 |
| **WP4** | Sections 12–15: signed SHAP + family attribution, caseload translation, fairness with declared threshold, timings. | Q12, Q13, Q14, Q16, GATE-3(R9) | WP2 | 1 |
| **WP5** | Repository hygiene: repo-relative paths, commit the diagnostic CSVs, reconcile `requirements.txt`/`environment.yml` against `pip_freeze.txt`, five-line reproduce recipe, tag the commit, resolve the orphan artefacts. | Q4, GATE-3(M19), Q26 | WP1 | 1 |
| **WP6** | **Freeze, commit, then run Section 16 once.** Score the test set. Re-score Table 6 from Section 17. | GATE-1(iii), Q19, Q25 | WP2–WP5 | 0.5 |
| **WP7** | Rewrite Method and Results against the new numbers, using Section 2 below for the number-independent parts and Section 4 for the tables. | GATE-3(3a), Q1, Q3, Q7, Q16, Q18 | WP6 | 2 |
| **WP8** | Theory redirection: read Finn and the household-investment literature; rewrite R6 at family level. | Q13, Q15 | D3, WP4 | parallel |

**Do not run WP6 before WP5.** Section 16 is gated on `FREEZE_CONFIRMED` and
a clean git commit precisely so that "test set scored once, after freeze,
committed" becomes a true sentence with a hash behind it.

---

## 2. Replacement wording — the fixes that need no new numbers

Drafted for you to edit, not to paste blind. Bracketed placeholders are
filled from the notebook's CSVs.

### M1 — add a design-justification paragraph (Q1, currently absent)

> The testbed must support one decision: whether substituting the training
> objective changes minority-class ranking, holding the substrate and its
> hyperparameters fixed. Three alternative designs were considered.
> A school-grouped cross-validation was rejected as a primary design because
> the sample contains only four school clusters, one of which supplies half
> the records; four clusters cannot support a grouped estimate, and this
> limitation is reported rather than concealed — a leave-one-school-out
> analysis is nonetheless presented as a bounded secondary check (R[x]).
> Nested cross-validation with an outer test fold was rejected because the
> research question fixes the hyperparameter configuration by design, so
> there is no inner selection loop for an outer fold to protect against;
> the single frozen holdout plus repeated cross-validation within the
> training partition achieves the same isolation at this sample size.
> A temporal holdout was unavailable: all records were collected within a
> four-week window and describe a single academic period.

### M6 — replace the leakage reassurance (GATE-1(i), GATE-3, Q7)

Delete the sentence beginning *"The corrected pipeline confirmed that the
direction and magnitude … were materially unchanged."* It is contradicted by
your own Table 5. Replace with:

> All results reported in this paper are generated by a pipeline in which
> every fitted transformation — median and mode imputation, categorical
> encoding, min-max scaling, near-constant column detection and all three
> composite constructions — is fitted on the training fold only and applied
> unchanged to the held-out fold. An earlier version of the pipeline fitted
> these transformations on the full dataset before partitioning. Under the
> corrected pipeline the *direction* of the baseline-versus-E-LightGBM
> difference is unchanged, but its magnitude and its significance both shift:
> at seed 42 the difference moves from [old] to [new] and the Wilcoxon p from
> [old] to [new]. Both figures are reported, and the corrected pipeline is
> the one that underwrites every table.

Add, in the same section:

> The column-reduction path from the 44 raw variables to the [n] final
> predictors is given in Table [x], with the criterion and the affected
> column names at each step. During remediation an error was found in the
> earlier identifier-removal rule: it tested for keyword *substrings* rather
> than whole tokens, so any column whose name contained the letters "id" —
> including [list from `columns_restored_vs_substring_rule.csv`] — was
> silently removed. The rule now matches whole underscore-delimited tokens
> and exact column names. The restored columns are present in the reported
> feature matrix, which accordingly differs from the R01 submission.

And the outlier statement:

> Attendance was recorded above its physical maximum in [n] instances across
> [k] terms (observed maximum [x]%). These values are clipped to 100% inside
> each training fold, using the fixed physical bound rather than a
> data-derived threshold, so the treatment introduces no information sharing
> across partitions. [n] records ([x]%) carried at least one missing value;
> per-column counts are in Table [x].

### M7 — print the governing constants (Q3, J11 breach)

> The attendance risk index is the weighted mean of (1 − attendance rate)
> across the three terms, with weights 0.25, 0.30 and 0.45 applied to terms
> one, two and three respectively, so that the most recent term carries the
> greatest weight. The socioeconomic vulnerability score is the mean of three
> components: LEAP beneficiary status (0/1), School Feeding Programme
> participation (0/1), and income vulnerability, defined as
> 1 − (income_ordinal / 2) where family income level is mapped to an explicit
> ordinal scale, Low = 0, Medium = 1, High = 2. Responses of "Don't know" are
> treated as missing and carry a separate indicator variable. The behavioural
> engagement index is the mean of (1 − scaled behaviour-warning count),
> scaled class participation and scaled extracurricular involvement, each
> min-max scaled on the training fold.
>
> An encoding fault in the R01 pipeline is corrected here. Family income
> level was previously label-encoded, which assigns integers alphabetically;
> on this data, and in the presence of a misspelled category ("Hgh"
> alongside "High"), the resulting ordering was Don't know, High, Hgh, Low,
> Medium, so the vulnerability component was not monotone in income. The
> misspelling is merged and the ordinal map above is applied explicitly.
> This changes the composite and therefore every number that used it.

### M9 — correct the clustering claim (GATE-1(iv), the flat contradiction)

Delete *"no school-level grouping variable spans partitions."* Replace:

> Each record represents a distinct pupil; no pupil appears more than once.
> The data do, however, carry a school-level grouping variable,
> `school_code`, with four unique values distributed [500 / 329 / 92 / 79]
> across the 1000 records, and school-level dropout rates ranging from 3.3%
> to 20.7%. All four schools appear on both sides of the pupil-level split.
> Because a pupil-level random split across four clusters would allow the
> model to learn cluster membership, `school_code` is excluded from the
> feature matrix in the primary analysis, and the claim is stated as
> pupil-level *within these four schools*. A leave-one-school-out analysis
> is reported separately (R[x]); with four clusters it is presented as a
> bounded robustness check, not as a generalisation estimate.

### M10 and M18 — correct the boundary (Q18; also fixes the tier)

Replace *"basic school pupils in the Kumasi Metropolitan Assembly"* in both
sections with:

> pupils in Primary 4 through JHS Form 2 at four basic schools in the Kumasi
> Metropolitan area, Ashanti Region, Ghana, with 500, 329, 92 and 79 records
> per school respectively.

The tier itself is correct and needs no change. **Also reconcile the academic
year**, which is currently stated three ways: Table 1 says 2024/25, M18 says
2025/2026, and M3/M4 give a collection window of 7 May – 1 June 2026. At most
one is right. Fix all four locations to the same value.

### M12 — soften the attribution claim (Q7)

Replace *"so that any observed difference can, in principle, be attributed to
the loss-function substitution alone"* with:

> The primary comparison holds the feature matrix, the hyperparameter
> configuration and the class-weighting mechanism identical between the two
> arms, so that the arms differ only in the `objective` argument. Contrasts
> that additionally vary the feature set or the weighting are reported
> separately in the ablation grid (Table [x]) and are labelled by the number
> of components they vary.

### M13 — disclose the search history (GATE-2, Q26; currently a flat contradiction)

Delete *"No automated hyperparameter search was conducted."* Replace:

> The primary comparison uses a single fixed configuration for both arms:
> 300 boosting rounds, 31 leaves, learning rate 0.05, row subsample 0.8,
> column subsample 0.8. Neither arm received any search trials, so tuning
> parity holds at zero trials each.
>
> Exploratory searches were conducted earlier in the project and are
> disclosed here in full. Notebook 6 ran `RandomizedSearchCV` with 40
> candidates for Random Forest, 40 for LightGBM and 15 for CatBoost; the
> CatBoost search used 3-fold cross-validation against 5 folds for the other
> two, which is an asymmetry in the exploratory phase and is noted as such.
> A separate 20-trial Optuna study was run on [date], recorded in
> `results/optuna_trials.csv`. The tuned configuration in
> `models/best_parameters.json` (819 estimators, 34 leaves, learning rate
> 0.0416) was not used for any reported result. A counted floor on the total
> number of model configurations fitted across the project, before folds, is
> [6 baselines + 24 augmentation configurations + 95 randomised-search
> candidates + 20 Optuna trials + N final variants]. No reported number was
> selected on the basis of these searches; the fixed configuration above was
> set before the comparison and held throughout.

### M15 — declare the fairness protocol and the threshold (Q16; R10 cites a Method that does not exist)

Add a subsection:

> **Fairness audit protocol.** Subgroup performance is assessed on two
> pre-specified attributes, pupil gender and geographic zone, using the
> Equal Opportunity difference (the maximum between-group gap in true
> positive rate) and the Equalized Odds difference (the larger of the
> between-group TPR gap and FPR gap), evaluated at the 0.5 decision
> threshold. A difference of 0.10 or below is treated as acceptable. This
> threshold is set by reference to [state your authority — a cited fairness
> guideline, an institutional standard, or a stated judgement about the cost
> of unequal detection across zones], and is fixed before the audit is run.
> Subgroup positive-case counts are reported alongside every difference,
> because a difference computed on few positives is not interpretable.

Also add the threshold disclosure:

> The decision threshold for all reported binary classifications is 0.5 and
> no threshold optimisation informs any reported result. A threshold of 0.19
> appears in `models/best_threshold.txt` from an exploratory run; it was not
> used for any reported number, and its caseload consequence is included in
> the threshold-to-caseload table (Table [x]) for completeness.

### M19 — regenerate the environment table (GATE-3)

Do not retype it. Paste `environment_versions.csv` from the notebook run, and
edit `requirements.txt` and `environment.yml` to agree with `pip_freeze.txt`
from the same run. Three files must say the same thing.

### M21 — add the locked run identifier

> The results reported here are generated by run `[RUN_ID]`, at commit
> `[hash]`, tagged `[tag]`. The run manifest, per-fold scores, per-instance
> predictions, environment capture and every derived table are committed
> under `results/remediation/[RUN_ID]/`.

### R2 — correct the identical-conditions claim (Q7)

Replace *"Both models were trained and evaluated under identical
conditions"* with:

> The two arms take the identical [n]-column feature matrix, the identical
> fixed hyperparameter configuration and the identical class-weighting
> mechanism; they differ only in the training objective supplied to
> LightGBM's `objective` parameter.

Also: **supply Supplementary Table S1 or delete the promise.** The six-model
comparison data exist in `results/baseline_results.csv` and
`baseline_ranking.csv`, so supplying it is the cheaper route and closes half
of objective 2.

### R5 — correct the ablation claim (Q20, another flat contradiction)

Delete *"Each row adds exactly one component relative to the row above."*
The step from row 2 to row 3 changed the loss **and** removed the imbalance
correction. Replace with the full grid from the notebook, and label each
pairwise delta with what it varies. Note in the text:

> `is_unbalance` is honoured only by LightGBM's built-in objectives and is
> inert under a custom objective. The R01 comparison therefore gave the
> baseline an explicit class-weighting correction and the proposed arm none
> beyond the focal α term. In the grid below, class weighting is implemented
> for both losses by the same mechanism — a balanced `sample_weight` — so
> that the weighting dimension is symmetric across arms.

### R6 — three fixes (Q13, Q14)

1. Delete *"above the remaining 38 raw features apart from one."* A feature
   ranked 3rd overall behind two named raw features cannot be above all but
   one of 38. Say: *ranked 3rd of 41, behind `average_exam_score` and
   `term_2_attendance`.*
2. Replace the top-five list with **family-level attribution across all
   features** (`shap_family_*.csv`), reporting summed mean |SHAP| per family
   and its share of total attribution.
3. Report **signed** attribution and state the direction of each top feature.
   Add: *"Attribution for the minority class rests on [n] positive instances
   in the explained partition; standard errors on the signed values are
   reported in [file]."* And label both SHAP artefacts by model —
   `shap_feature_importance.csv` is the baseline (38 features),
   `feature_importance_focal.csv` is the proposed model (41). A reader
   comparing them currently finds the top feature differing by a factor of
   two with no explanation.

### R9 — regenerate, then keep your own caveat

Replace the timings with `training_times.csv` from the locked run. Keep the
existing sentence about not being able to attribute the difference to the
focal computation from these measurements — the examiner singled it out as
the register the rest of the manuscript needs.

### Objectives — restructure to one primary, two supporting (Q16)

Objective 1 as written is a data-collection step, and the ablation answered it
negatively without the text saying so. Objective 3 bundles four activities.
Suggested restructure:

- **Primary:** engineer a focal-loss training objective for LightGBM and
  determine whether substituting it for binary cross-entropy changes
  minority-class ranking performance, holding the feature matrix,
  hyperparameters and class weighting fixed.
- **Supporting 1:** construct three domain-informed composite features and
  measure their contribution to ranking performance under ablation.
  *(Report the answer as negative where it is negative.)*
- **Supporting 2:** evaluate the resulting models with AUC-PR translated to
  an operational caseload, a signed SHAP attribution analysis, a
  pre-specified subgroup fairness audit, and paired significance testing
  with effect size and power reported.

Each now terminates in a named table. Then build the traceability matrix.

---

## 3. What is permanently unfixable, and belongs in Limitations

Write these yourself, before a reviewer writes them for you.

1. **The old test partition was scored repeatedly.** Across Notebooks 4, 5b,
   6, 6b, 7 and 8. No re-run fixes the history of a partition. State it: the
   R01 numbers came from a partition that had been scored multiple times; the
   R02 numbers come from a single scoring after a committed freeze.
2. **Four schools, one holding half the sample.** Not closable by adding
   pupils to the same four schools.
3. **Statistical power.** MDE 0.0131 against an observed effect near 0.0018.
   Your own power analysis found this, which is creditable — keep the
   "inconclusive" wording, which is what earns clearance condition C3.
4. **The metric has no headroom.** A baseline at 0.992 AUC-PR making one
   error on 200 cases leaves an intervention nothing to move. Even a
   fully repaired null would certify that *this testbed cannot resolve the
   question*, not that focal loss does not help. Those two statements must be
   separated explicitly in the text.

And the examiner's sharpest point, which is worth more than the null:

> A model reaching 0.995 accuracy with a single error on 200 pupils, from
> administrative and questionnaire data at a 9.2% base rate, does not look
> like any real dropout-prediction problem in the literature. What is
> carrying that separation?

`school_code` at 10th in the SHAP ranking, across four schools with dropout
rates from 3.3% to 20.7%, is the first place to look — and dropping it (D1)
plus the leave-one-school-out analysis (Section 10) is exactly the test.
**If performance falls substantially once `school_code` is removed or once a
whole school is held out, that is your paper.** It is a far more interesting
contribution than a null on a loss function.

---

## 4. Collection window — act now, not later

Ethics approval (HuSSREC/AP/543/VOL. 5) runs to **30 June 2027**, so the
window is open. Causes 4, 12 and 17 are all detected and all close with more
*schools* — not more pupils in the same four. This is the single
highest-value action available: recruiting additional sites closes Q2, part
of GATE-1, part of Q23 and Q22 simultaneously. After submission it becomes a
permanent limitation.

Raise it with your supervisor this week, even if the answer is no. Being able
to write "recruitment of additional sites was considered and was not feasible
within [constraint]" is worth much more than silence.

---

## 5. Notes on the notebook

`Notebook_10_Remediation.ipynb` writes everything under
`results/remediation/<RUN_ID>/` with a manifest, so every reported number
traces to one run identifier — which is what GATE-3 asks for.

**Before your first run:**

- Every column name is in one config cell (Section 1, third code cell).
  Expect to fix some on the first pass; nothing else needs editing.
- `ORDINAL_MAPS` currently contains only `family_income_level`. **Every other
  genuinely ordered questionnaire variable you leave out of it is still being
  label-encoded alphabetically** — parental education band, travel distance
  band, any Likert item stored as text. Go through the codebook and add them.
- `FEATURE_FAMILIES` drives the Q13 family attribution. Check the keyword
  lists against your real column names or features will land in
  `other_unclassified`.
- `FAIRNESS_THRESHOLD = 0.10` must be written into the Method *before* you
  read Section 14.
- Set `SCORE_TEST = False` for the whole exploratory pass. Section 16 refuses
  to run without both flags and a git commit.

**Honest caveat on the code:** it was written from your notebooks, not
executed against your data — I do not have `cleaned_data.csv`. I smoke-tested
the preprocessing, the grid plumbing, the contrast maths, the variance
decomposition, the instance-level stratification and the caseload table
against synthetic data matching your schema, with LightGBM stubbed out (it is
not installable in my sandbox). The logic runs end to end. Expect to fix
column names, and read Section 2's output carefully on the first pass — if
the restored-column list is non-empty, your feature matrix has genuinely
changed and every previously reported number moves.

One thing to verify yourself on the first real run: that
`is_unbalance` is in fact inert under a custom objective, as R5's replacement
wording above asserts. Fit `G_all_focal_noW` and `H_all_focal_W` with
`is_unbalance` instead of `sample_weight` and check whether the predictions
are identical. If they are, the claim stands and it is worth a sentence,
because it means the R01 baseline had a class-weighting correction the
proposed arm never received.
