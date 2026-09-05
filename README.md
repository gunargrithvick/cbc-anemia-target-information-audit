# Anemia detection on NHANES CBC data — leakage audit and leakage-free benchmark

Companion code for *"How much of anemia-detection accuracy is hemoglobin
measuring itself?"* — a study of arithmetic target leakage in CBC-based
anemia classification.

## The problem in one paragraph

Anemia is defined by a hemoglobin threshold, so hemoglobin cannot be a
legitimate input to an anemia classifier. Papers therefore drop it. That is
where the trouble starts, because a haematology analyser does not measure six
red-cell numbers. It measures **three** — RBC, HGB, HCT — and **divides** to
print the other three:

```
MCV  = HCT / RBC * 10        mean cell volume
MCH  = HGB / RBC * 10        mean cell hemoglobin
MCHC = HGB / HCT * 100       mean cell hemoglobin concentration
```

So the count is a subtraction: **six published variables − three exact defining
equations = three degrees of freedom.** Nothing in that sentence is estimated
from the data. These identities are not empirical regularities that happen to
hold in NHANES; they are the formulas the analyser evaluated in order to have a
sixth column to print at all, which is why they hold to the last published digit
rather than merely correlating.

Rearranging the same three equations puts hemoglobin back. Four minimal routes
exist, and `identity_audit.py` scores all four on the cohort:

```
HGB = MCH  * RBC / 10          needs {MCH, RBC}          MAE 0.030 g/dL
HGB = MCHC * HCT / 100         needs {MCHC, HCT}         MAE 0.030
HGB = MCHC * MCV * RBC / 1000  needs {MCHC, MCV, RBC}    MAE 0.030
HGB = MCH  * HCT / MCV         needs {MCH, HCT, MCV}     MAE 0.032
```

Each agrees with the WHO anemia label on 99.3–99.4% of cases. A feature set that
retains any one of those four groups has not dropped hemoglobin — it has renamed
it. The fourth route is the one that is easy to miss, because it needs neither
RBC nor MCHC, so a check that looks for the obvious pairs will pass it.
`MCV * RBC / 10` does the same thing for hematocrit (MAE 0.038 %). That is how a
feature list can look clean and still be leaky.

The rank test in `identity_audit.py` is a **check** on that count of three, not
the source of it, and reading it requires one detail: the identities are
products, so they are linear only after taking logs. On the standardised log
block, 3 components carry 99.9% of the variance — the arithmetic count. The same
test on levels needs 4, because a linear method cannot see a product. That extra
component is a limitation of the test, not a fourth degree of freedom. Both
numbers are reported, because the levels answer is what a reader gets by default
and it is the wrong one.

## What this adds

1. A **9-rung leakage ladder** (L0–L7, including a demographics variant of the
   submitted feature set) measuring what each retained identity is worth, with
   Hb recoverability scored as best-of-family — exact identity route, RF, OLS,
   OLS in logs — where the family member is chosen on the **validation** block
   and then scored once on test, so the test rows select nothing. The oracle
   that selecting on test would have given is reported alongside, as
   `best_on_test`, together with the difference it would have made.
2. A **2×2 decomposition** separating the feature-side leak (identity retained)
   from the label-side one (a single 12.0 g/dL cut makes the label a function
   of Hb alone). Reported on the lift scale, accuracy minus each cell's own
   no-information rate, because changing the label also moves the class
   balance and raw accuracies are not comparable across cells.
3. **Design-based inference**: the NHANES survey design (24 strata, 49 PSUs,
   25 df) with the Rao–Wu rescaled bootstrap, against which the naive cluster
   bootstrap is shown to be too narrow. Two estimands are kept apart, and each
   JSON says which one it reports: `ablation.json` and `benchmark.json` give
   **unweighted within-sample** contrasts with cluster-aware uncertainty —
   does rung A beat rung B on these rows — while `weighted.json` gives
   **population** quantities with `WTMECPRP` as the base weight. Weighting the
   contrasts would change the question, not sharpen the answer.
4. **Out-of-cycle replay** of the ladder on three cycles the models never saw.

## Data

Four NHANES cycles, public domain, no registration, eight files:

```bash
python src/download_data.py
```

| cycle | CBC | demographics | MEC weight | role |
|---|---|---|---|---|
| 2017–Mar 2020 | `P_CBC.XPT` | `P_DEMO.XPT` | `WTMECPRP` | primary; the only cycle used weighted |
| 2015–2016 | `CBC_I.XPT` | `DEMO_I.XPT` | `WTMEC2YR` | out-of-cycle replication |
| 2013–2014 | `CBC_H.XPT` | `DEMO_H.XPT` | `WTMEC2YR` | out-of-cycle replication |
| 2011–2012 | `CBC_G.XPT` | `DEMO_G.XPT` | `WTMEC2YR` | out-of-cycle replication |

Weights, strata and PSUs are cycle-specific — stratum 145 in 2011–12 is not
stratum 145 in 2015–16 — so the cycles are never pooled and handed to one
variance estimator. Every design-based estimate stays inside 2017–2020; the
earlier cycles are used unweighted, as replication samples.

## Layout

```
anemia_detection/
├── src/
│   ├── paths.py         project paths, resolved from __file__
│   ├── download_data.py four cycles, and why they must not be pooled
│   ├── dataprep.py      cohort, WHO labels, 9-rung ladder, 60/20/20 split,
│   │                    survey design arrays, Rao-Wu weights
│   ├── identity_audit.py    the analyser identities and the rank deficiency
│   ├── ablation.py      the ladder, the 2x2 decomposition, Holm-corrected
│   │                    contrasts, train-size sensitivity, out-of-cycle replay
│   ├── benchmark.py     7 estimators + 2 trivial baselines, CIs, PR-AUC,
│   │                    calibration, validation-selected thresholds
│   ├── weighted.py      design-based prevalence and weighted model metrics
│   ├── figures.py       the ten paper figures, read from the JSON
│   ├── anemia_app.py    the deployable L6 screening tool
│   └── run_all.py       one command for the whole study + run manifest
├── data/                eight NHANES XPT files (gitignored)
├── models/              anemia_binary.pkl, anemia_severity.pkl,
│                        model_info.json, rejections.json (gitignored)
├── results/             identity_audit, ablation, benchmark, weighted,
│                        importance, run_manifest .json (tracked) +
│                        run_all.log and run_all.prev.log (gitignored)
├── figures/             fig1…fig10 as .png (300 dpi) and .pdf (gitignored)
├── tests/               nine test files; see Tests below
├── pytest.ini
├── requirements.txt
└── README.md
```

Paper material lives outside the project, in `../manuscript/`: the manuscript
extract, reference verification output, text extracts of the 20 reviewed
papers, and the one-off reference-audit scripts.

Scripts resolve their own paths, so both of these work:

```bash
python src/dataprep.py
cd src && python dataprep.py
```

## Reproducing

```bash
pip install -r requirements.txt
python src/run_all.py
```

It runs the eight stages in dependency order, times each one, checks that each
stage actually wrote the artefacts it claims, stops at the first failure, and
writes `results/run_manifest.json` — interpreter, library versions, seed,
per-stage timings, and a SHA-256 of every NHANES input, every `src/*.py` plus
`requirements.txt`, and every result JSON, figure and model pickle. That manifest
is what makes a reproduction checkable rather than asserted. Full output is teed
to `results/run_all.log`.

The source hashes are the leg that is easy to leave out. Matching inputs and
matching outputs still allow two runs to have executed different code, which is
the state this project was found in once — every stage green, every JSON valid,
and the manifest four hours older than `ablation.py`. They are taken *before* the
first stage, so a file edited while the long pipeline run is in flight cannot be
recorded as the code that produced results it never touched; `run_all.py` hashes
again at the end and writes the comparison down as
`sources_unchanged_during_run`.

Three files are deliberately not hashed, each with its reason in the manifest
under `artefacts_not_hashed`: `model_info.json` records its own training time,
`models/rejections.json` is a tally the interactive app appends to, and
`figures/app_overview.png` is written by `anemia_app.py` menu option 3 rather than
by any stage — it is the one image in `figures/` that no figure table lists.
Hashing a file the pipeline does not control makes the manifest disagree with the
disk for reasons that are not results, and a provenance check that cries wolf is
one people stop reading.

Flags: `--list`, `--only <stage>`, `--from <stage>`, `--skip <stage>`,
`--force-download`.

The stages are also runnable on their own, in this order:

```bash
python src/download_data.py
python src/dataprep.py
python src/identity_audit.py
python src/ablation.py
python src/benchmark.py
python src/weighted.py
python src/figures.py
python src/anemia_app.py
```

`figures.py` runs last because every number it prints is read from the four
result JSONs, not recomputed, so no figure can disagree with a table. It does
refit models in one place — `figures.refit()` at L6 — because PR curves,
reliability curves and confusion matrices need per-row predictions that no JSON
stores. It imports `benchmark.models()` rather than redefining the estimators,
and the refit reproduces `benchmark.json` **bit-for-bit**: checked on all nine
model entries (seven estimators plus two trivial baselines), with maximum absolute
difference in test accuracy and PR-AUC of 0.0. It also
writes `results/importance.json` (MDI vs permutation importance at L6).

## Tests

```bash
python -m pytest tests -q
python -m pytest tests -q -m "not slow"
python -m pytest tests -q --cov=src --cov-report=term-missing
python quality_check.py
```

The third command measures source coverage and is a diagnostic rather than a
pipeline stage. The current test suite reaches 24.1% of executable source lines:
the unit and contract tests exercise the safety-critical rules, while the
long-running analysis stages are reproduced separately by `run_all.py`.
Coverage output files are ignored because they are local test diagnostics, not
research results.

`quality_check.py` is the one-command fast check. It runs the strict test suite
with coverage, static name checks, and dependency consistency checks. It does
not rerun the full research pipeline; after changing source code or pinned
dependencies, run `python src/run_all.py` first and then run the quality check.

A pipeline run proves the code executed. These prove the things no single run can
check, because a run that is internally consistent with stale code looks exactly
like a correct one — which is the state this project was found in once, with the
manifest four hours older than `ablation.py`.

| file | what it pins |
|---|---|
| `test_identities.py` | the three analyser identities, against a **per-row rounding bound** derived from NHANES' own print precision rather than a guessed tolerance; all four `HGB_ROUTES`; that a route reproduces the WHO *label*, not just the number |
| `test_labels.py` | every WHO band as published; `nan` for the four uncovered cases (under 6 months, fractional ages in band gaps, a third sex code, missing input); the pregnancy override applying only to women 15+; the flat-12 cut wrong for 6,895 people and label-changing for 540 |
| `test_split_and_design.py` | 7,293/2,431/2,432 disjoint, exhaustive, deterministic, stratified so no block is empty of the 26 severe cases; 24 strata / 49 PSUs / 25 df; Rao-Wu weights constant within a cluster and `MECWT`-free when `base=None`; all **five** branches of `dp.t_and_p`, including the non-finite input that used to be published as p = 0.0 and the zero-df case that returns `NaN` from `stats.t.sf`; and that the split does **not** separate clusters, which is documented, not accidental |
| `test_ladder.py` | each rung's leakage class re-derived from `HGB_ROUTES`, including every *subset* of the identity-free rungs; that L4–L6 still rebuild hematocrit, which is the limitation the paper has to state |
| `test_download_checks.py` | 200 record-aligned truncations of `P_CBC.XPT` all refused — the exact class the old three-test check accepted, including the half-file its docstring cites; 182/200 still refused under a filename `EXPECT` does not know; and that a single flipped byte is **not** caught, asserted so the non-claim cannot quietly become a claim |
| `test_results_contract.py` | every `src/*.py` and `requirements.txt` still hashing to what the manifest recorded, and none of them edited mid-run; every artefact and input still hashing too; nothing in `figures/` that the manifest neither hashes nor names as an exclusion; every JSON key the figures dereference, listed by hand so a rename fails a test instead of a figure; no bare `NaN` in `results/` |
| `test_app_guards.py` | `PROMPTS` aligned to the model's columns by *name*; pregnant-male and out-of-window pregnancy refused; ages 43–44 flagged as extrapolation rather than refused; and the eight ways the guard tally can be incoherent, including the hand-edited string that used to crash the tool instead of quarantining the file |

Tests marked `slow` read the NHANES files or a `results/*.json`; they skip
cleanly if an artefact has not been built yet. Nothing writes inside `data/`,
`models/` or `results/` — the truncation and tally tests work on copies in
pytest's `tmp_path`.

## Figures

| file | content |
|---|---|
| `fig1_flowchart` | data → labels → one split → two models → evaluation (R6-3) |
| `fig2_ladder` | accuracy and PR-AUC across the ladder, over Hb recoverability |
| `fig3_decomposition` | features × label definition, on the lift scale |
| `fig4_pr_curves` | precision–recall for the seven real estimators at L6 (the two Dummy baselines are the dashed no-skill line, not curves), with the four validation-selected operating points |
| `fig5_calibration` | reliability diagram and Brier scores |
| `fig6_confusion` | Random Forest confusion matrices, L1 vs L6, same test rows |
| `fig7_importance` | MDI vs permutation importance, and where they disagree |
| `fig8_prevalence` | sample vs population prevalence by subgroup, design summary |
| `fig9_contrasts` | the six pre-declared contrasts under Holm, and the three variance scales side by side |
| `fig10_out_of_cycle` | the ladder replayed on 2011–12, 2013–14 and 2015–16 |

## Numbers

One cohort definition and one split, shared by every script, so nothing can
drift between tables. `SEED = 42`, in `dataprep.py`.

**Cohort.** 13,772 CBC records → 12,156 complete on all features (11.73%
excluded). Split 60/20/20 into 7,293 train / 2,431 validation / 2,432 test,
stratified on the 4-class severity label. Every model in the paper is fitted on
the training block only; the validation block selects estimators and
thresholds; the test block is scored once.

**Labels.** WHO age-, sex- and pregnancy-specific hemoglobin cut-offs
(11.0 / 11.5 / 12.0 / 12.0 / 13.0 g/dL), not a single 12.0 g/dL cut. A flat
12.0 g/dL is the wrong threshold for **6,895 of 12,156 people (56.72%)** —
888 belong at 11.0, 1,556 at 11.5, 4,451 at 13.0, and only 5,261 at 12.0.
Prevalence 9.60% unweighted, 6.63% survey-weighted (SE 0.45). Majority-class
no-information rate 90.40% — the floor every accuracy in the paper is quoted
against. None of the 20 reviewed papers reports a majority-class or
no-information baseline for its anemia classification, so none of their reported
accuracies can be read against chance. That is a claim about this one
comparator, measured by keyword search over the 20 text extracts; it is not a
claim that those papers report no baselines of any kind.

**Identity audit** (`results/identity_audit.json`). All three identities hold to
the published reporting grid on 97.7% of records; the 279 that do not (2.30%)
miss by more than propagated rounding can explain and are analyser or
transcription errors in those records, not evidence against the identity. Of the
31 non-empty subsets of the five non-hemoglobin indices, **16 reconstruct
hemoglobin to MAE < 0.05 g/dL**, and the split is not a knife edge: the worst
"exact" subset is at 0.0314 g/dL and the best non-exact one at 0.2395, a gap of
0.21 g/dL, so any cutoff inside it gives the same 16. The smallest exact routes
need two variables each — `{HCT, MCHC}` and `{RBC, MCH}`. Those 31 subset MAEs
are **in-sample** log-linear least-squares fits, scored on the same rows they
were fitted on, and `identity_audit.json` says so in its own `fit_note`: this
audit asks whether hemoglobin is *arithmetically present* in a subset, not
whether a model would generalise, so an in-sample fit is the right instrument
and an out-of-sample one would understate it. The four closed-form routes in the
same JSON (`routes.to_hgb`) have no fitted parameter at all, and the in-sample
subset fits rediscover them: `{HCT, MCHC}` comes back with exponents 0.9996 and
0.9992, `{RBC, MCH}` with 0.9995 and 1.0000, which is the identity being
recovered rather than a curve being bent to the data. The honest out-of-sample
version of the question is the ladder's
`recoverability` block, which fits on train, selects on validation and scores on
test. Variance in the last
three log directions is 0.0226% observed against 0.0072% for exact values
rounded onto the same grid (3.13×); with the 279 inconsistent records dropped it
falls to 0.0087% against a null rebuilt on those same rows (1.19×). Dropping 279
rows *at random* instead leaves it at 0.0228% (3.16×), which is the control that
rules out sample size as the explanation.

**Ladder** (binary accuracy, test block, against a 0.9042 floor):

| rung | features | accuracy |
|---|---|---|
| L0 | Hb present verbatim | 0.9548 |
| L1 | the submitted set, RBC MCV MCH | 0.9502 |
| L1d | L1 + age, sex, pregnancy | 0.9819 |
| L4 | identity-free CBC | 0.9354 |
| L6 | identity-free CBC + age, sex, pregnancy | 0.9667 |
| L7 | Hb + the WHO cut-off inputs | 1.0000 |

Recoverability is the best-of-family MAE with the family member picked on the
validation block and scored once on test. `selection_optimism_gdl` — how much
picking the winner on test would have added — is **0.0000 g/dL on all nine
rungs**: validation and test choose the same estimator every time, so the
honest rule costs nothing here. It is still the rule, because that had to be
measured rather than assumed, and the shortcut's bias would have run toward this
paper's own conclusion. `ols in logs` wins on every learned rung, which is what
the identities being products predicts.

**Raw values are screened, not cleaned.** `dataprep.integrity_screen` checks all
109,404 published CBC values against containment bounds wider than any reference
interval. Three fall outside — two MCVs at 35.4 fL and one WBC at 400 ×10³/µL —
and all three are real extreme phenotypes, not errors, so **no row is dropped**:
a plausibility filter would move the denominator of every number in the paper on
the basis of the analyst's idea of a normal patient, and would remove exactly
the patients a screening tool exists to catch. Zero non-positive values, zero
non-finite values, zero duplicate `SEQN`. Complete-case is the only exclusion in
the study, and its bias is quantified rather than hidden.

**Decomposition** (lift points, each cell against its own no-information rate).
Retaining the identity is worth 3.29 points under the old sex-blind `Hb < 12`
label and 1.48 under the WHO label; correcting the label is worth 5.30 points
with the identity present and 3.49 with it gone; interaction 1.81; the two
together account for 6.78 of the submitted result's lift. The two sources are
not additive, which is exactly why the 2×2 is needed and why comparing L1 with
L6 directly — changing both at once — comes out the wrong sign. This block signs
its numbers the other way round from the contrasts block below — here positive
means *the leaky choice scores higher*, because the quantities are named as costs
and benefits rather than as transitions — so `identity_effect_WHO` is +0.0148 in
`decomposition` and −0.0148 in `contrasts`. Both JSON blocks carry their own
`sign_convention` string; the magnitudes and *p*-values are identical.

**Inference.** Six contrasts, pre-declared, Holm step-down. Every delta below is
signed as `ablation.json` signs it — accuracy(after) − accuracy(before), the
sign of the transition the contrast's own name describes — so a negative number
means the second rung scores lower. Five contrasts survive; the null is
`L0 (Hb verbatim) → L1 (the submitted set)` at **−0.0045**, Holm *p* = 0.184 —
dropping hemoglobin for the identity route costs 0.45 accuracy points and that
is indistinguishable from zero, so the submitted feature set is statistically
indistinguishable from handing the model hemoglobin. Variance from 1000 Rao–Wu
rescaled bootstrap replicates on 25 design df. For unweighted *paired* accuracy
differences the Rao–Wu SE is 0.90× the i.i.d.-row SE (median), while the naive
cluster bootstrap gives 0.645× — narrower than ignoring the design entirely,
which is the tell that it is biased downward by (n_h−1)/n_h. The design effect
of 4.00 belongs to the weighted prevalence, not to these paired differences;
treating it as a universal divisor overstates the correction.

**Sensitivity.** One conclusion depends on the fit rule: `L1 → L4` (remove the
identity, no demographics) is **−0.0148** (*p* = 0.002) fitted on the 60% block
and **−0.0086** (*p* = 0.149) fitted on train+validation — same sign, same
direction, but only the first clears 0.05. It is reported as fragile rather than
dropped. The companion contrast `L1d → L6` holds in both arms (−0.0152,
*p* < 0.001; −0.0164, *p* = 0.0001).

**Survey design.** Weighted prevalence 6.63% (SE 0.45), DEFF 4.00, effective
*n* ≈ 3,039. That 4.00 decomposes into **2.417 from unequal weighting** (Kish
1+CV², weight CV 1.190) **× 1.654 from clustering** — so clustering alone widens
an interval by 1.29×, not 2.00×, and quoting DEFF as if it were all clustering
overstates the design's cost. The unweighted out-of-fold L6 accuracy of 0.9630
falls *outside* the population interval [0.9674, 0.9782]; the other four metrics
fall inside. Weighting the *fit* changes 1.538% of predictions and lowers
population accuracy by 0.0039 (Holm *p* = 0.003 across the family of three; raw
*p* = 0.0009), so the unweighted fit is kept as a
defended choice, not an oversight: weights make estimators unbiased for
population quantities, they do not make a classifier better at classifying.

**Out-of-cycle.** The L1−L4 identity gap is positive in all three replication
cycles, but ranges from +0.23 to +1.77 lift points. The claim the data supports
is that the gap reappears, not that it is worth a fixed amount.

## Known limits, stated here rather than buried

- **Complete-case exclusion is not MCAR.** The 1,616 dropped rows (11.73%)
  average 17.7 years against 37.5 for the 12,156 kept — a standardised mean
  difference of **−0.873**, large by any convention, with 71.8% of the dropped
  under 18 against 30.0% of the kept. Age is one of the variables that sets the
  WHO cut-off, so the loss is non-random in the label as well as in the
  covariate. The survey weights correct for the sample design; they do not
  correct for CBC non-response.
- **The i.i.d. row split is not a cluster split.** `dataprep.split3` splits rows,
  not PSUs, so test rows can share a PSU with training rows. `ablation.py`
  measures what that buys: mean absolute optimism 0.0017, maximum 0.0059, all
  six contrast signs preserved, largest contrast movement 0.0065.
- **`RIDEXPRG` is collected only for women 20–44.** Coding blanks as
  non-pregnant assigns them the higher cut-off, so reported prevalence is an
  upper bound for that group.
- **MDI is biased** toward high-cardinality features; `fig7` shows three rank
  flips against permutation importance at L6, so MDI alone is not reported.
- The 5×4 repeated CV inside each script is i.i.d. over rows and is a stability
  check only. Every interval quoted as an interval is design-based.

## The screening tool

```bash
python src/anemia_app.py
```

Uses the L6 rung: RBC, MCV, RDW, age, sex, pregnancy. It does **not** ask for
hemoglobin — that defines the label, so a tool that received it would not be
predicting anything. Test-block accuracy 0.9667 binary and 0.9515 for the
3-class severity model, both fitted with 300 trees on the same 60% training
block as everything else so `models/model_info.json` is comparable with the
paper. Input guards are two-layer: per field (numeric, in range, whole where a
fraction is not a value) and then cross-field, because every per-field check
passes on `sex = 1` with `pregnant = 1`. That combination is now rejected, as is
`pregnant = 1` outside ages 20–44 — the only ages at which NHANES ascertains
pregnancy through `RIDEXPRG`, and so the only ages at which the training data
contains a row coded pregnant (all 76 of them fall between 20 and 42). The guard
follows the ascertainment window, not biology: pregnancies outside it exist and
NHANES simply does not record them, so a rejection here means "this model was
never shown such a row", not "this cannot happen". Every rejection is logged to
`models/rejections.json` under the field that caused it, so the rejection rate
can be reported instead of asserted.
