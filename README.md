# CBC Anemia Target-Information Audit

Reproducible NHANES analysis of whether removing the hemoglobin (HGB) column
removes hemoglobin-related target information from complete blood count (CBC)
feature sets.

## Research Question

Anemia labels are defined from hemoglobin. Several red-cell indices reported in
a CBC are related to hemoglobin through analyzer equations:

```text
MCV  = 10 × HCT / RBC
MCH  = 10 × HGB / RBC
MCHC = 100 × HGB / HCT
```

Therefore, deleting the HGB column does not by itself demonstrate that
hemoglobin-related information has been excluded. This repository audits that
property rather than proposing a new classifier or a new clinical relationship.
The analysis distinguishes deterministic recoverability from the broader claim
of target leakage. In particular, a performance difference between two rungs is
not treated as a pure leakage estimate when the rungs also contain different
physiological variables.

## Scope and Contributions

- A nine-entry feature ladder covering direct HGB, analyzer-derived routes,
  identity-reduced CBC features, and demographic variants.
- Four closed-form HGB reconstruction routes and learned reconstruction checks.
- Seven classifier families: logistic regression, RBF SVM, decision tree,
  Random Forest, Extra Trees, Histogram Gradient Boosting, and XGBoost, plus
  majority and stratified baselines.
- A fixed train/validation/test split: 60%/20%/20%, stratified by the four-class
  severity label, with random seed 42.
- Validation-only model and threshold selection followed by one test evaluation.
- Paired contrasts with Holm correction and Rao--Wu rescaled bootstrap
  uncertainty using the NHANES survey design.
- Label-definition, training-size, PSU-held-out, and out-of-cycle sensitivity
  analyses.

## Data Sources

The project uses official public-use files from the [CDC/NCHS NHANES
program](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020).
The primary analysis uses the 2017--March 2020 pre-pandemic files:

| Cycle | CBC file | Demographics file | Weight | Role |
|---|---|---|---|---|
| 2017--March 2020 | `P_CBC.XPT` | `P_DEMO.XPT` | `WTMECPRP` | Primary analysis |
| 2015--2016 | `CBC_I.XPT` | `DEMO_I.XPT` | `WTMEC2YR` | Out-of-cycle replication |
| 2013--2014 | `CBC_H.XPT` | `DEMO_H.XPT` | `WTMEC2YR` | Out-of-cycle replication |
| 2011--2012 | `CBC_G.XPT` | `DEMO_G.XPT` | `WTMEC2YR` | Out-of-cycle replication |

The primary merge contains 13,772 records. Complete-case processing retains
12,156 records and excludes 1,616 records with required values missing. The
earlier cycles are replayed separately and are not pooled with the primary
cycle because their survey design variables are cycle-specific.

Raw NHANES files are downloaded locally by `src/download_data.py` and are not
included in the repository. The download script checks the expected XPORT
structure, file size, and record count before accepting a file.

## Repository Structure

```text
src/
  download_data.py     Download and validate the NHANES files
  dataprep.py          Cohort, labels, feature ladder, split, survey design
  identity_audit.py    Analyzer identities and HGB recoverability
  ablation.py          Ladder, decomposition, contrasts, sensitivities
  benchmark.py         Classifier comparison and evaluation metrics
  weighted.py          Survey-weighted prevalence and model metrics
  figures.py           Reproducible result figures
  anemia_app.py        L6 screening demonstration
  run_all.py           Full pipeline and provenance manifest

tests/                 Unit and contract tests
results/               JSON evidence and run manifest
figures/               Generated figures and the paper workflow source
models/                Locally generated models; ignored by Git
requirements.txt       Pinned runtime and test dependencies
quality_check.py       Fast tests, static checks, and dependency check
```

Manuscript files and literature-review materials are maintained separately and
are not part of this code repository.

## Reproducibility

From the repository root:

```bash
pip install -r requirements.txt
python src/run_all.py
python quality_check.py
```

`run_all.py` executes the stages in dependency order, records their status and
timings, and writes `results/run_manifest.json`. The manifest records the
Python and library versions, random seed, hashes of the eight input files,
source files, result JSON files, generated figures, and model pickles. It also
names files that are intentionally not hashed because they are generated
outside the numerical pipeline or change during interactive use.

Individual stages can also be run in this order:

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

The full pipeline is computationally heavier than `quality_check.py`. The
quality check does not rerun the research pipeline; after changing source code
or dependencies, run the full pipeline first.

## Main Results

The current JSON results report:

- Four analyzer-based HGB reconstruction routes with mean absolute errors from
  0.0302 to 0.0315 g/dL and WHO-label agreement from 99.325% to 99.441%.
- Random Forest accuracy of 0.9502 for the MCH-containing submitted rung and
  0.9354 for the identity-reduced CBC rung, a difference of 1.48 percentage
  points on the primary test block.
- A positive L1--L4 accuracy difference in all three earlier NHANES cycles,
  with the magnitude varying from 0.23 to 1.77 percentage points.
- Survey-weighted prevalence, design-based confidence intervals, and paired
  uncertainty estimates based on the primary NHANES design.

The 1.48-point difference is not interpreted as a pure leakage effect. L1 and
L4 differ in both target-information recoverability and physiological feature
content. The results therefore support an audit of recoverability and model
behavior, not a universal estimate of accuracy inflation.

## Label Definitions and Evaluation Design

The primary labels use age-, sex-, and pregnancy-specific WHO hemoglobin
cut-offs. The severity analysis uses four WHO bands and merges the moderate and
severe classes because the severe group is small. A separate sensitivity
analysis evaluates the manuscript's single 12.0 g/dL cut.

The main split is row-wise, so records from the same primary sampling unit (PSU)
can occur in different partitions. PSU-held-out evaluation quantifies the
effect of this choice. The earlier NHANES cycles are out-of-cycle replication,
not independent clinical external validation.

## Tests and Quality Checks

```bash
python -m pytest tests -q
python -m pytest tests -q -m "not slow"
python -m pytest tests -q --cov=src --cov-report=term-missing
python quality_check.py
```

The current suite checks analyzer identities, label construction, feature-rung
definitions, split and survey-design properties, download-file integrity,
result contracts, application input guards, static names, and dependency
consistency. The latest automated source coverage is approximately 24%; this
is a software-quality measure and does not replace reproduction of the full
research pipeline.

## Known Limitations

- Complete-case exclusion is non-random with respect to age and may introduce
  selection bias.
- The primary row-wise split permits PSU sharing between partitions.
- The NHANES analysis is not external clinical validation.
- The screening demonstration is a research tool and is not a medical device or
  a substitute for laboratory diagnosis.

## Screening Demonstration

```bash
python src/anemia_app.py
```

The demonstration uses the identity-reduced L6 feature set: RBC, MCV, RDW, age,
sex, and pregnancy. It does not request HGB because HGB defines the target in
this study. Input guards reject invalid combinations and record rejections in
the locally generated `models/rejections.json` file.

## License

The code is released under the MIT License. NHANES data are obtained from the
CDC/NCHS public-use files linked above; users should consult the official NHANES
documentation and analytic guidance when reusing the data.
