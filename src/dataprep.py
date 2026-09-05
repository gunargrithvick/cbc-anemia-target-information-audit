"""
Single source of truth for the NHANES anemia analysis.

Everything downstream (ablation.py, benchmark.py, weighted.py, identity_audit.py,
figures.py, anemia_app.py, run_all.py) imports from here so that the cohort, the
labels and the train/validation/test split cannot drift between scripts.

What this module owns
  1. loading and merging CBC + DEMO, for any of the four cycles
  2. the exclusion cascade, with counts, and what the exclusions are made of
  3. WHO age/sex/pregnancy-specific hemoglobin thresholds
  4. the binary and severity labels derived from those thresholds
  5. the feature-set ladder: nine rungs, crossing how much analyser identity is
     left in the features against whether demographics are present
  6. one shared stratified train / validation / test split
  7. the survey-design arrays every variance estimate needs

Two things here are deliberate and easy to misread as bugs.

  The split is three-way. Model choice is made on the validation part and the
  test part is touched once, at the end, by every script that uses split3. The
  earlier version of this study chose its headline estimator by comparing
  test-set scores, which is the same optimism it accuses other papers of.
  weighted.py's population block does not use split3 at all - see split3's
  docstring for why, and weighted.json for what it costs.

  The primary cohort is 2017-March 2020 and nothing is pooled across cycles.
  NHANES weights, strata and PSUs are cycle-specific, so pooling them and
  handing the result to one variance estimator would be wrong. The other three
  cycles are replication samples, used unweighted.

Run directly to print the cohort description:  python dataprep.py
"""

import math
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from download_data import CYCLES, PRIMARY, cycle_files, weight_col

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
TEST_SIZE = 0.20          # of the cohort
VAL_SIZE = 0.25           # of what is left, so 60 / 20 / 20 overall

# The two hyperparameters that have to agree across scripts, here because they
# did not.
#
# N_TREES was written as a literal in ablation.py, benchmark.py, weighted.py and
# anemia_app.py. The app's copy said 400 while the ladder's said 300, and the two
# published a severity accuracy of 0.9502 and 0.9515 for what both described as
# the same model - a discrepancy that took a reader comparing model_info.json
# against ablation.json to find, and the first fix was to re-type 300 in the
# fourth place rather than to stop having four places. N_FOLDS was four literals
# for the same reason.
#
# These are the values, not defaults: a script that wants a different forest
# should say so at the call site and say why, not carry a private constant that
# happens to differ.
N_TREES = 300
N_FOLDS = 5

# The replicate count for the Rao-Wu rescaled bootstrap, here for the same
# reason and after the same kind of discrepancy.
#
# ablation.py used 1000 and benchmark.py and weighted.py used 500, all three on
# the same design, the same seed and - for the L6 binary test accuracy - the same
# 2,432 test rows. So the SAME design SE was published twice with two values:
# 0.00407619 in ablation.json and 0.00408166 in benchmark.json. Neither is wrong;
# the difference is Monte-Carlo noise in the replicate spread. But a reader who
# lines the two files up finds two numbers for one quantity and has no way to
# learn from either file that the replicate count is why, and a reviewer who
# finds it first is entitled to ask what else disagrees.
#
# 1000, the larger of the two, because the cost is a weighted mean over 2,432
# rows per replicate and the benefit is that the two files now agree exactly.
N_REP = 1000

# ---------------------------------------------------------------- variables --
HGB = "LBXHGB"        # hemoglobin  g/dL   -- LABEL SOURCE. A feature only on
                      # L0 and L7, the two rungs whose purpose is to show what
                      # handing the model its own label is worth; never a
                      # feature of any model this project would deploy.

RBC = "LBXRBCSI"      # red cell count     10^6/uL
HCT = "LBXHCT"        # hematocrit  %
MCV = "LBXMCVSI"      # mean cell volume   fL
MCH = "LBXMCHSI"      # mean cell hemoglobin  pg
MCHC = "LBXMC"        # mean cell hemoglobin concentration  g/dL
RDW = "LBXRDW"        # red cell distribution width  %
PLT = "LBXPLTSI"      # platelets  10^3/uL
WBC = "LBXWBCSI"      # white cells  10^3/uL

# Containment bounds: outside them a value is a transcription or unit error,
# inside them it is a patient. Deliberately wider than any reference interval,
# and deliberately wider than the observed cohort range - see integrity_screen,
# which reports against these and drops nothing.
#
# This lives at module level because anemia_app.py needs the SAME numbers. The
# app used narrower "plausibility" ranges of its own (RBC 2.0-6.5, MCV 60-110,
# RDW 10-30) and those refused 49 real cohort rows, 34 of them from the very
# training block its forest was fitted on, and 23 of the 49 anemic - a 46.9%
# anemia rate against the cohort's 9.60%. So the deployed tool was rejecting
# precisely the patients it exists to catch, while this file's own policy string
# argued against doing that. One definition now, in one place.
CONTAINMENT_BOUNDS = {
    HGB: (3.0, 25.0), RBC: (1.0, 8.0), HCT: (10.0, 70.0),
    MCV: (40.0, 140.0), MCH: (10.0, 50.0), MCHC: (20.0, 45.0),
    RDW: (8.0, 40.0), PLT: (5.0, 1500.0), WBC: (0.5, 100.0),
}

# canonical names for the three design columns, because the weight column is
# called WTMEC2YR in the 2-year cycles and WTMECPRP in the pre-pandemic file
MECWT = "MECWT"
STRATUM = "SDMVSTRA"
PSU = "SDMVPSU"

DEMO_RAW = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDEXPRG", STRATUM, PSU]

# demographics that DEFINE the WHO threshold. A model given hemoglobin plus
# these can reconstruct the label exactly -- see LADDER rung L7.
AGE = "RIDAGEYR"
SEX = "RIAGENDR"
PREG = "PREGNANT"          # derived, 0/1, never missing
DEMO_FEATURES = [AGE, SEX, PREG]
PREG_OK = "PREG_ASCERTAINED"   # derived, 0/1: did NHANES actually ask?

# Columns that are CODES, not quantities. RIAGENDR is 1 or 2 and the 2 does not
# mean twice the 1; PREGNANT is a 0/1 indicator.
#
# Here because ablation.hb_recoverability's log-linear fit logs every strictly
# positive column, and on that rule RIAGENDR qualifies - so the JSON was
# publishing a coefficient on log(sex) under the key `is_exponent: true`, i.e.
# "hemoglobin is proportional to sex to the power -0.0121". No such quantity
# exists. Removing SEX from the log cannot move a single fitted value, because
# the log of a two-valued column is an affine function of that column and spans
# the same one-dimensional space, so the fit is identical and only the reported
# interpretation changes; PREGNANT contains zeros and was never logged. The
# published recoverability figures are therefore unaffected, which is why this
# was a labelling bug and not a numerical one.
NOMINAL_FEATURES = [SEX, PREG]

# ------------------------------------------------------------ feature sets --
# The rungs of the ladder. Two things vary along it, and they are varied
# separately so that no comparison confounds them:
#
#                          no demographics        with age, sex, pregnancy
#   identity present       L1_paper_leaky         L1_paper_leaky_demo
#   identity absent        L4_identity_free       L6_identity_free_demo
#
# L1 -> L4 is the price of removing the arithmetic route to the label.
# L1 -> L1d and L4 -> L6 are the value of demographics the CBC does not carry.
# Comparing L1 with L6 directly, as the first version of this study did, changes
# both at once and comes out with the wrong sign.
#
# The analyser identities that make the leak possible:
#     MCV  = HCT / RBC * 10        ->  HCT = MCV  * RBC / 10
#     MCH  = HGB / RBC * 10        ->  HGB = MCH  * RBC / 10
#     MCHC = HGB / HCT * 100       ->  HGB = MCHC * HCT / 100
# so any set containing {MCH, RBC} or {MCHC, HCT} or {MCHC, MCV, RBC} or
# {MCH, HCT, MCV} reconstructs the label to within ~0.03 g/dL. That fourth set
# is the one that is easy to miss: HGB = MCH * HCT / MCV needs neither RBC nor
# MCHC. Note the first identity too: {MCV, RBC} rebuilds hematocrit, so "we
# dropped HCT" is not the same claim as "the model cannot see HCT".
#
# HGB_ROUTES is the authoritative list, used by describe()'s cleanliness tagger.
# identity_audit.py verifies it two ways: each route in closed form, and an
# exhaustive search over all 31 subsets of the five non-hemoglobin indices.
HGB_ROUTES = [
    frozenset({MCH, RBC}),
    frozenset({MCHC, HCT}),
    frozenset({MCHC, MCV, RBC}),
    frozenset({MCH, HCT, MCV}),
]
LADDER = {
    # hemoglobin handed to the model directly -- the trivial upper bound
    "L0_hgb_direct":  [HGB, RBC, MCV, MCH],
    # the submitted paper's feature set: MCH x RBC / 10 = HGB
    "L1_paper_leaky": [RBC, MCV, MCH],
    # the same leaky set plus the demographics, so the 2x2 above is complete
    "L1_paper_leaky_demo": [RBC, MCV, MCH, AGE, SEX, PREG],
    # MCH dropped, but MCHC x HCT / 100 = HGB
    "L2_mchc_hct":    [RBC, MCV, MCHC, HCT],
    # every one of the four HGB_ROUTES is now broken -- L3 keeps neither MCH nor
    # MCHC, and all four routes need one or the other. HCT is still here, and
    # {MCV, RBC} would rebuild it anyway, which is why L3 is not a step down from
    # L2 in information terms -- see identity_audit.py.
    "L3_hct_only":    [RBC, MCV, HCT],
    # free of every exact route to hemoglobin: no member of {HGB, HCT, MCH, MCHC}
    "L4_identity_free": [RBC, MCV, RDW],
    # identity-free plus non-red-cell CBC channels
    "L5_identity_free_plus": [RBC, MCV, RDW, PLT, WBC],
    # identity-free features PLUS the demographics a clinician legitimately
    # knows. This is the defensible model: no EXACT route to hemoglobin
    # survives. It is not information-free -- a least-squares fit in logs on
    # L6's own six columns still predicts hemoglobin to 0.246 g/dL and reproduces
    # the WHO label on 97.7% of held-out rows, which ablation.py measures rung by
    # rung and reports as `ols in logs`. "Identity-free" is the claim;
    # "uninformative about hemoglobin" is not, and would be false.
    #
    # Those two numbers are L6's, from rungs.L6_identity_free_demo.
    # recoverability.selected in results/ablation.json (0.246224 and 0.976974).
    # They used to read 0.249 and 97.4%, which are L4's - the right order of
    # magnitude and the wrong rung, and the manuscript's limitation sentence
    # about the DEPLOYED model quotes this comment. Adding demographics makes
    # hemoglobin slightly MORE recoverable, not less, so the L4 figures
    # understated the very thing the sentence concedes.
    "L6_identity_free_demo": [RBC, MCV, RDW, AGE, SEX, PREG],
    # hemoglobin plus the variables that define its own threshold. Nothing is
    # being predicted here: the label is a deterministic function of the input.
    "L7_hgb_demo": [HGB, AGE, SEX, PREG],
}

# columns required for the shared cohort. Demographic features are derived or
# always present, so they must NOT drive the dropna.
#
# This is deliberately the UNION over the whole ladder, including L5's platelet
# and white-cell channels, which only L5 uses. The reason is comparability: a
# different cohort per rung would make the rungs incomparable, and comparing
# rungs is the entire study.
#
# This comment used to add that requiring PLT and WBC "costs 1,616 rows that L6
# could otherwise have kept". That was false, and ablation.complete_case_
# sensitivity is what measures it. The CBC is missing PERSON BY PERSON, not
# analyte by analyte: of 13,772 merged records, 12,156 carry all nine values and
# 1,616 carry none of them - none of the 21 CBC columns in the file, not just
# none of the nine. So every rung's own maximal complete-case cohort IS this one,
# exactly 12,156, and the union requirement costs nobody anything. The
# comparability is free.
#
# That makes the complete-case attrition a property of the survey rather than of
# this feature list, which is the stronger form of the limitation: no choice of
# rung reduces it, and dropping L5 would not buy back a single row.
ALL_FEATURE_COLS = sorted(
    {c for cols in LADDER.values() for c in cols if c.startswith("LBX")} | {HGB})

SEVERITY_NAMES_3 = ["Normal", "Mild", "Moderate-Severe"]
SEVERITY_NAMES_4 = ["Normal", "Mild", "Moderate", "Severe"]


# ------------------------------------------------------------- WHO cut-offs --
def who_thresholds(age, sex, pregnant):
    """
    WHO haemoglobin cut-offs, g/dL (WHO/NMH/NHD/MNM/11.1).

    group                        normal    mild        moderate    severe
    6-59 months, and pregnant    >= 11.0   10.0-10.9   7.0-9.9     < 7.0
    children 5-11 y              >= 11.5   11.0-11.4   8.0-10.9    < 8.0
    children 12-14 y             >= 12.0   11.0-11.9   8.0-10.9    < 8.0
    non-pregnant women 15+       >= 12.0   11.0-11.9   8.0-10.9    < 8.0
    men 15+                      >= 13.0   11.0-12.9   8.0-10.9    < 8.0

    Returns three arrays: (normal_cut, moderate_cut, severe_cut). Anything no WHO
    band covers returns nan rather than being folded into the nearest band. Four
    kinds of input land there, and the fourth is the only one that can happen in
    these data:

      - under 6 months, including a row recorded as age 0, for whom WHO publishes
        no cut-off at all;
      - a fractional age inside a gap between two whole-year bands: 4.5 falls
        between 6-59 months and children 5-11, 11.5 between 5-11 and 12-14, 14.5
        between 12-14 and adult. The bands are written on whole years because
        RIDAGEYR is whole years, so these gaps are unreachable from NHANES and
        reachable from a hand-entered row;
      - an adult whose sex is neither 1 nor 2, since the adult cut-off is sex-
        specific and RIAGENDR has no third code to fall back on;
      - a missing age or sex, which propagates: nan compares false everywhere, so
        no band claims the row.

    nan is the honest answer in all four cases and it is not silently absorbed:
    _label drops rows whose threshold is nan, and anemia_app rejects the input
    before it gets this far.

    The submitted manuscript applied a single 12.0 g/dL cut to everyone. How many
    people that is the wrong cut-off for is not a number to quote from memory:
    main() prints the WHO_NORMAL distribution and how many rows are not at 12.0.
    """
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    preg = np.asarray(pregnant, dtype=float)

    normal = np.full(age.shape, np.nan)
    moderate = np.full(age.shape, np.nan)
    severe = np.full(age.shape, np.nan)

    # WHO's lowest band is 6-59 months, so it starts at 0.5 years, not at 0.
    # RIDAGEYR is whole years: a row recorded as 0 could be under 6 months, for
    # whom WHO publishes no cut-off, and `age <= 4` alone handed it the infant
    # 11.0 g/dL cut anyway. NHANES draws no CBC below age 1, so this excludes
    # nobody in this cohort; it stops the tool from inventing a threshold if a
    # future cycle or a hand-entered row ever reaches age 0.
    young = (age >= 0.5) & (age <= 4)
    mid = (age >= 5) & (age <= 11)
    teen = (age >= 12) & (age <= 14)
    adult = age >= 15
    preg_now = adult & (sex == 2) & (preg == 1)

    # young children and pregnant women share the lower band structure
    for mask in (young, preg_now):
        normal[mask], moderate[mask], severe[mask] = 11.0, 10.0, 7.0

    normal[mid], moderate[mid], severe[mid] = 11.5, 11.0, 8.0
    normal[teen], moderate[teen], severe[teen] = 12.0, 11.0, 8.0

    women = adult & (sex == 2) & ~preg_now
    normal[women], moderate[women], severe[women] = 12.0, 11.0, 8.0

    men = adult & (sex == 1)
    normal[men], moderate[men], severe[men] = 13.0, 11.0, 8.0

    return normal, moderate, severe


def pregnancy_flags(df):
    """
    Pregnancy as a feature, plus an honest record of where it is a guess.

    RIDEXPRG is 1 = pregnant, 2 = not pregnant, 3 = could not ascertain, and it
    is only collected for women aged 20-44. Everyone else is blank. Coding blank
    as "not pregnant" is what every paper does, including this one, but it is an
    assumption and it is not free: the pregnant cut-off is 11.0 g/dL against
    12.0 for a non-pregnant woman, so a pregnant woman coded non-pregnant is
    pushed toward the anemic label.

    PREGNANT          the feature, 1 only for a recorded RIDEXPRG == 1
    PREG_ASCERTAINED  1 where RIDEXPRG is 1 or 2, i.e. where the answer is real
    """
    preg = (df["RIDEXPRG"] == 1).astype(int)
    ascertained = df["RIDEXPRG"].isin([1.0, 2.0]).astype(int)
    return preg, ascertained


# ------------------------------------------------------------------- cohort --
def _label(df):
    """
    Attach WHO thresholds and every label column. Returns a NEW frame.

    It does not mutate the caller's: df.assign copies, and the dropna below
    copies again, so the argument comes back untouched and the return value is
    the only thing worth keeping. load_cycle and cohort_on both reassign.
    """

    normal, moderate, severe = who_thresholds(df[AGE], df[SEX], df[PREG])
    df = df.assign(WHO_NORMAL=normal, WHO_MODERATE=moderate, WHO_SEVERE=severe)
    df = df.dropna(subset=["WHO_NORMAL"]).copy()

    hb = df[HGB].to_numpy()
    # binary: WHO-correct, and the manuscript's sex-blind version for comparison
    df["Anemia"] = (hb < df["WHO_NORMAL"].to_numpy()).astype(int)
    df["Anemia_hb12"] = (hb < 12.0).astype(int)

    # four-class WHO severity, and the manuscript's merged three-class version
    df["Severity4"] = np.select(
        [hb >= df["WHO_NORMAL"], hb >= df["WHO_MODERATE"], hb >= df["WHO_SEVERE"]],
        [0, 1, 2], default=3).astype(int)
    df["Severity"] = np.minimum(df["Severity4"], 2)
    return df


def who_threshold_audit(df):
    """
    Which WHO cut-off each person got, and how wrong one flat 12.0 g/dL is.

    describe() has printed this to stdout since the first version and no artefact
    recorded it, so the single number the manuscript's label criticism rests on -
    how many people a sex-blind Hb < 12 cut is wrong for - existed only in a
    console log. figures.py then re-typed the five cut-offs into a flowchart box
    as a literal string, which is the same fact stated a third time in a place
    nothing checks.

    Returns the threshold histogram, the count and share the flat cut misclassifies
    the threshold for, and how many binary labels actually flip.
    """
    n = len(df)
    thr = df["WHO_NORMAL"]
    wrong = int((thr != 12.0).sum())
    flip = int((df["Anemia"] != df["Anemia_hb12"]).sum())
    return {
        "thresholds_gdl": sorted(float(v) for v in thr.unique()),
        "rows_per_threshold": {f"{float(t):.1f}": int(c)
                               for t, c in thr.value_counts().sort_index().items()},
        "n_rows": n,
        "rows_whose_cutoff_is_not_12": wrong,
        "pct_whose_cutoff_is_not_12": round(100.0 * wrong / n, 2) if n else None,
        "binary_labels_that_flip_under_a_flat_12_cut": flip,
        "pct_binary_labels_that_flip": round(100.0 * flip / n, 2) if n else None,
        "note": ("the flat cut is wrong about the THRESHOLD for far more people "
                 "than it is wrong about the LABEL: most people are nowhere near "
                 "either cut-off. Both counts are reported because quoting only "
                 "the first overstates the damage and quoting only the second "
                 "understates the error."),
    }


def merged_raw(cycle=PRIMARY):
    """
    CBC joined to demographics, BEFORE any completeness filter.

    Returns (raw, weight_column, counts), where counts holds the two input file
    sizes so a caller building an attrition cascade does not have to re-read the
    XPT files to get them.

    This is the frame every cohort in the project is carved out of, and it is
    exposed because two different carvings are needed: load_cycle's, which
    requires completeness on the UNION of all ladder features so the rungs are
    comparable, and cohort_on's, which requires it on one rung's own columns so
    the cost of that union can be measured.

    Extracted from load_cycle rather than duplicated, so there is exactly one
    definition of the merge and one place a rename can go wrong.
    """
    cbc_path, demo_path = cycle_files(cycle)
    wt = weight_col(cycle)
    cbc = pd.read_sas(cbc_path, format="xport")
    demo = pd.read_sas(demo_path, format="xport")
    demo = demo[DEMO_RAW + [wt]].rename(columns={wt: MECWT})
    counts = {"cbc_records": int(len(cbc)), "demo_records": int(len(demo))}
    return cbc.merge(demo, on="SEQN", how="inner"), wt, counts


def required_columns(cols):
    """
    The columns a row must carry for one rung to be able to use it.

    The rung's own measured channels, plus HGB: without hemoglobin there is no
    label to score against, so a row missing it is not a row this rung could have
    used either. Demographic features are derived or always present, so they must
    not enter the dropna.

    Its own function because ablation.complete_case_sensitivity needs the same
    rule on the replication cycles, where it does not want a cohort built - and
    it had the `{c for c in cols if c.startswith("LBX")} | {HGB}` expression
    written out a second time, which is one edit away from the primary cycle and
    the replication cycles answering the same question differently.
    """
    return sorted({c for c in cols if c.startswith("LBX")} | {HGB})


def cohort_on(cols, cycle=PRIMARY, raw=None):
    """
    The cohort ONE rung could have had: complete on its own columns only.

    ALL_FEATURE_COLS makes every rung share a cohort, which costs rows that a
    rung not using PLT or WBC never needed to lose. That trade is stated in this
    file's comments; this function is what lets ablation.py measure it instead of
    asserting that it is cheap.

    Returns (df, required_columns). The label source HGB is always required -
    without it there is no label to score against, so a row missing HGB is not a
    row this rung could have used either.

    Pass `raw` to reuse one merge across all nine rungs.
    """
    need = required_columns(cols)
    if raw is None:
        raw, _, _ = merged_raw(cycle)
    df = raw.dropna(subset=need).copy()
    df[PREG], df[PREG_OK] = pregnancy_flags(df)
    df = _label(df)
    return df.reset_index(drop=True), need


def load_cycle(cycle=PRIMARY, verbose=False):
    """
    Build the analysis cohort for one NHANES cycle.

    Returns (df, attrition). The attrition dict carries the cascade counts AND a
    comparison of the rows that were dropped against the rows that were kept,
    because "11.7% were excluded for incomplete CBC" is only reassuring if the
    11.7% look like the rest of the sample, and here they do not.
    """
    raw, wt, counts = merged_raw(cycle)
    att = {"cycle": cycle, "weight_column": wt, **counts}
    att["after_demo_merge"] = int(len(raw))

    df = raw.dropna(subset=ALL_FEATURE_COLS).copy()
    att["complete_on_all_features"] = int(len(df))

    df[PREG], df[PREG_OK] = pregnancy_flags(df)
    df = _label(df)
    att["with_who_threshold"] = int(len(df))

    # One percentage used to stand for the whole cascade, which works only while
    # the demographics merge and the WHO banding lose nobody. They are separated
    # so a future cycle that does lose rows there cannot have the loss read as
    # incomplete CBC.
    att["excluded_at_demo_merge"] = (att["cbc_records"]
                                     - att["after_demo_merge"])
    att["excluded_incomplete_features"] = (att["after_demo_merge"]
                                           - att["complete_on_all_features"])
    att["excluded_no_who_band"] = (att["complete_on_all_features"]
                                   - att["with_who_threshold"])
    att["excluded_total"] = att["cbc_records"] - len(df)
    att["excluded_pct"] = round(100.0 * att["excluded_total"] / att["cbc_records"], 2)
    att["n_analysis"] = int(len(df))
    att["attrition_profile"] = attrition_profile(raw, df)
    att["who_thresholds"] = who_threshold_audit(df)
    att["pregnancy"] = pregnancy_audit(df)
    att["integrity"] = integrity_screen(df)

    df = df.reset_index(drop=True)
    if verbose:
        describe(df, att)
    return df, att


def load_cohort(verbose=False):
    """The primary cohort, 2017-March 2020. Kept as a name for compatibility."""
    return load_cycle(PRIMARY, verbose=verbose)


def load_replication(verbose=False):
    """
    The three earlier cycles, as separate cohorts.

    Not concatenated. Their weights, strata and PSUs are not comparable with
    each other or with the primary file, so anything that pools them would be
    producing a variance estimate for a design that does not exist.
    """
    out = {}
    for cycle in CYCLES:
        if cycle == PRIMARY:
            continue
        df, att = load_cycle(cycle)
        out[cycle] = (df, att)
        if verbose:
            print(f"  {cycle}  n = {len(df):,}  "
                  f"anemia {df.Anemia.mean() * 100:.2f}%  "
                  f"(excluded {att['excluded_pct']}%)")
    return out


def attrition_profile(raw, kept):
    """
    Are the excluded rows exchangeable with the kept rows?

    Complete-case analysis is only harmless under MCAR. Age is not just any
    covariate here: it is one of the variables that sets the WHO cut-off, so
    dropping rows non-randomly in age drops them non-randomly in the label.
    This returns the comparison rather than asserting anything about it.
    """
    dropped = raw.loc[~raw["SEQN"].isin(set(kept["SEQN"]))]
    if not len(dropped):
        return {"n_dropped": 0}

    def prof(d):
        age = d[AGE].to_numpy(dtype=float)
        return {"n": int(len(d)),
                "mean_age": round(float(np.nanmean(age)), 2),
                "sd_age": round(float(np.nanstd(age, ddof=1)), 2),
                "median_age": round(float(np.nanmedian(age)), 1),
                "pct_under_2": round(float(np.nanmean(age < 2) * 100), 2),
                "pct_under_18": round(float(np.nanmean(age < 18) * 100), 2),
                "pct_65_plus": round(float(np.nanmean(age >= 65) * 100), 2),
                "pct_female": round(float(np.nanmean(
                    d[SEX].to_numpy(dtype=float) == 2) * 100), 2)}

    d, k = prof(dropped), prof(kept)
    gap = d["mean_age"] - k["mean_age"]

    # The same loss on the population scale, which is the scale weighted.py
    # reports on and the one where the omission was doing damage. weighted.py
    # publishes "population represented 292.9 million" from the kept rows'
    # weights; the dropped rows carry weight too, and nothing said how much. It
    # is 26.0 million, 8.1% of what the merged frame represents - so every
    # population total in weighted.json is a complete-case total that understates
    # the frame by that much, and the anemia counts by roughly +0.8 to +1.7
    # million depending on whether the dropped rows carry the kept prevalence or
    # the prevalence their age profile implies. Reported here, once, next to the
    # unweighted attrition, because these are the same exclusion.
    #
    # The dropped rows have no CBC at all, hence no label, so the anemia burden
    # among them cannot be measured - only bounded. That is why this records the
    # population lost and not an imputed count.
    if MECWT in raw.columns:
        w_all = float(pd.to_numeric(raw[MECWT], errors="coerce").fillna(0).sum())
        w_drop = float(pd.to_numeric(dropped[MECWT], errors="coerce")
                       .fillna(0).sum())
        weighted_loss = {
            "population_in_merged_frame": w_all,
            "population_dropped": w_drop,
            "population_dropped_pct": (round(100.0 * w_drop / w_all, 2)
                                       if w_all else None),
            "note": ("every population total in weighted.json is a "
                     "complete-case total over the kept rows; this is what the "
                     "excluded rows would have added. They carry no CBC, so "
                     "their anemia burden can be bounded but not measured."),
        }
    else:
        weighted_loss = {"note": "no weight column in the merged frame"}

    # A standardised difference, not a threshold on years and not a p-value.
    # This field used to read mcar_plausible: abs(gap) < 2.0 - an unsourced cut
    # on an unstandardised scale, under a name that claimed something no
    # comparison of two means can establish. Equal means do not make data MCAR;
    # they only fail to contradict it on the one variable compared. A p-value
    # would be worse still: at n = 13,772 a tenth of a year is significant.
    sd_pool = float(np.sqrt((d["sd_age"] ** 2 + k["sd_age"] ** 2) / 2.0))
    smd = round(gap / sd_pool, 3) if sd_pool else None
    return {"n_dropped": d["n"], "dropped": d, "retained": k,
            "age_gap_years": round(gap, 2),
            "age_standardised_difference": smd,
            "smd_convention": "|d| 0.1 small, 0.2 notable, 0.5 large",
            "weighted_loss": weighted_loss,
            "mcar_contradicted_on_age": (None if smd is None
                                         else bool(abs(smd) >= 0.1)),
            "mcar_note": "a difference here rules MCAR out; the absence of one "
                         "does not rule it in",
            "note": "age sets the WHO cut-off, so non-random loss in age is "
                    "non-random loss in the label"}


def integrity_screen(df):
    """
    Is any published CBC value impossible, rather than merely unusual?

    This does NOT clean anything, and that is the finding. The obvious next step
    after a complete-case filter is a plausibility filter, but a plausibility
    filter on a 12,156-row cohort silently changes the denominator of every
    number in the paper, and it does it on the basis of the analyst's idea of a
    normal patient. So the screen runs, the count is published, and the rows
    stay.

    The limits are containment bounds, deliberately wider than any reference
    interval: outside them a value is a transcription or unit error, inside them
    it is a patient. They live in CONTAINMENT_BOUNDS at module level so that
    anemia_app.py's input guards use the same numbers rather than a private,
    narrower set. Measured on 2017-2020, three values of 109,404 fall outside
    - two MCVs at 35.4 fL and one WBC at 400 x10^3/uL - and all three are
    clinically real (severe microcytosis; leukocytosis at presentation), so
    dropping them would remove exactly the patients a screening tool exists to
    catch. Zero non-positive values, zero non-finite values, zero duplicate
    SEQNs.

    That adjudication is about those three values and does not generalise, which
    matters because load_cycle runs this screen on all four cycles. The policy
    string used to assert that whatever falls outside the bounds is "a real
    extreme phenotype, not an error" for every cycle it ran on, and that is false
    on 2013-2014: SEQN 77520 carries MCHC 69.6 g/dL, above any attainable
    intracellular hemoglobin concentration, and RBC 1.67 with MCH 74.5 to match.
    The three analyser identities hold on that row to within rounding, so the
    identity check does not catch it either - the row is internally consistent
    and externally impossible, which is what a unit or transcription error looks
    like. It is still not dropped, because this screen drops nothing in any cycle;
    it is listed. outlying_values below names every flagged value with its SEQN so
    a reader adjudicates them rather than taking a blanket claim on trust.

    identity_audit.py refuses to run at all on a non-positive or non-finite
    value, because it logs and divides by these columns. That guard is the hard
    one; this is the soft report.
    """
    limits = CONTAINMENT_BOUNDS
    per_col, flagged, outliers = {}, 0, []
    for col, (lo, hi) in limits.items():
        v = df[col].to_numpy(dtype=float)
        bad = (v < lo) | (v > hi)
        out = int(bad.sum())
        flagged += out
        for i in np.flatnonzero(bad):
            outliers.append({"seqn": int(df["SEQN"].to_numpy()[i]),
                             "column": col, "value": float(v[i]),
                             "containment_bounds": [lo, hi]})
        per_col[col] = {"min": round(float(v.min()), 3),
                        "max": round(float(v.max()), 3),
                        "containment_bounds": [lo, hi],
                        "outside_bounds": out,
                        "non_positive": int((v <= 0).sum()),
                        "non_finite": int((~np.isfinite(v)).sum())}
    return {"n_rows": int(len(df)),
            "n_values_screened": int(len(df) * len(limits)),
            "values_outside_containment_bounds": flagged,
            "outlying_values": outliers,
            "non_positive_values": sum(c["non_positive"] for c in per_col.values()),
            "non_finite_values": sum(c["non_finite"] for c in per_col.values()),
            "duplicate_seqn": int(df["SEQN"].duplicated().sum()),
            "rows_dropped_by_this_screen": 0,
            "policy": "report, do not clean. A plausibility filter would move "
                      "the denominator of every number in the paper on the "
                      "basis of the analyst's idea of a normal patient, and in "
                      "the primary cycle the values it would remove are real "
                      "extreme phenotypes - two MCVs at 35.4 fL and one WBC at "
                      "400 x10^3/uL. The complete-case filter is the only "
                      "exclusion in this study, and its bias is quantified in "
                      "attrition_profile.",
            "adjudication": "being outside these bounds does not make a value "
                            "real, and being inside them does not make it right. "
                            "The primary cycle's three were checked one at a "
                            "time; outlying_values names every flagged value with "
                            "its SEQN so the same can be done for any cycle. "
                            "2013-2014's SEQN 77520 (MCHC 69.6 g/dL, RBC 1.67, "
                            "MCH 74.5) is not a phenotype - it is internally "
                            "consistent across all three analyser identities and "
                            "still physically impossible, which is what a unit "
                            "error looks like. It is retained, because this "
                            "screen drops nothing, and disclosed here.",
            "by_column": per_col}


def pregnancy_audit(df):
    """
    How much of the pregnancy feature is a recorded answer, and what it costs.

    The flip count is computed by CALLING who_thresholds twice - once with the
    coded flag and once with pregnant = 1 - instead of hard-coding an age window.
    The window version over-counted. It used women aged 12-49, but
    who_thresholds gives the pregnant band only from age 15, so for a girl of
    12-14 the label cannot move whatever the flag says: 31 rows were reported as
    flippable that this module's own thresholds cannot flip, making the published
    count 105 where it should have been 74. Deriving it from the function means
    the two definitions cannot disagree again.

    What is left is a real disclosure rather than a patched number. WHO states the
    11.0 g/dL pregnancy cut-off with no lower age bound, so requiring age >= 15
    for it under-covers pregnancy in girls of 12-14. That changes no row in these
    data - RIDEXPRG is collected only at ages 20-44, so PREGNANT is never 1 below
    20 - but it is a property of who_thresholds rather than of the data, and
    guessed_rows_with_no_pregnant_band below counts who sits there.
    """
    fem = df[SEX] == 2
    repro = fem & df[AGE].between(12, 49)
    guessed = repro & (df[PREG_OK] == 0)

    # every guessed row is coded non-pregnant, i.e. given the HIGHER cut-off.
    # If any of them are in fact pregnant their cut-off drops, which can only
    # turn an anemic label off. So the reported prevalence is an upper bound for
    # this group, and this is the size of that bound - measured against the
    # threshold who_thresholds would actually assign, not against a literal 11.0.
    age = df[AGE].to_numpy(dtype=float)
    sex = df[SEX].to_numpy(dtype=float)
    cut_if_pregnant, _, _ = who_thresholds(age, sex, np.ones(len(df)))
    cut_as_coded, _, _ = who_thresholds(age, sex, df[PREG].to_numpy(dtype=float))
    hb = df[HGB].to_numpy()
    g = guessed.to_numpy()
    anemic = df["Anemia"].to_numpy() == 1
    would_flip = int((g & anemic & (hb >= cut_if_pregnant)).sum())
    no_band = int((g & (cut_if_pregnant == cut_as_coded)).sum())
    return {"recorded_pregnant": int((df["RIDEXPRG"] == 1).sum()),
            "recorded_not_pregnant": int((df["RIDEXPRG"] == 2).sum()),
            "could_not_ascertain_code3": int((df["RIDEXPRG"] == 3).sum()),
            "women_12_49": int(repro.sum()),
            "women_12_49_not_ascertained": int(guessed.sum()),
            "labels_that_could_flip_off": would_flip,
            "flip_pct_of_all_positives": round(
                100.0 * would_flip / max(int(df["Anemia"].sum()), 1), 2),
            "guessed_rows_with_no_pregnant_band": no_band,
            "flip_definition": "guessed, currently anemic, and with hemoglobin at "
                               "or above the cut-off who_thresholds would assign "
                               "the same row if it were coded pregnant",
            "no_pregnant_band_note": "rows where coding the flag on would not move "
                                     "the cut-off at all, so no flip is possible: "
                                     "girls 12-14, for whom this implementation "
                                     "keeps the 12.0 band. WHO sets no lower age "
                                     "bound on the pregnancy cut-off, so that is "
                                     "an implementation choice, and it is inert "
                                     "here because RIDEXPRG is collected only at "
                                     "ages 20-44.",
            "direction": "reported prevalence is an upper bound for this group"}



# -------------------------------------------------------------------- split --
def split3(df):
    """
    ONE stratified 60 / 20 / 20 split, shared by every model and every rung.

    Stratified on the four-class severity rather than the binary label, because
    Severity4 is a refinement of Anemia: stratifying on it stratifies both, and
    the severe class is only a couple of dozen people in the primary cycle, so
    leaving it to chance is how a test fold ends up with none. The stratifying
    variable is fixed rather than a parameter: a different one would give a
    different split, and "one split shared by everything" has to be a guarantee
    to be worth anything.

    train       fit
    validation  every choice: which estimator, which threshold, any tuning
    test        touched once, for the numbers that go in the paper - by the
                scripts that use this split, which is all of them except
                weighted.py's population block (see below)

    There is deliberately no "refit on train + validation" helper. Every model
    that uses this split is fitted on the train block alone, which is what makes
    ablation.py, benchmark.py, figures.py and anemia_app.py comparable to each
    other. ablation.py's train_size_sensitivity builds the 80% set itself, in one
    place, to report what that choice would change.

    Two callers do NOT use this split, and saying "every model in this project"
    was wrong about both. weighted.py's model_oof block takes 5-fold out-of-fold
    predictions over all 12,156 rows, because its estimand is a population
    quantity that needs the whole design rather than a fifth of it; the tradeoff
    is stated and measured in weighted.json's model_oof.shared_cluster_optimism.
    ablation.cluster_holdout deliberately splits on PSUs instead of rows, which is
    the point of it. Neither is a bug and neither is covered by the sentence
    above, so the sentence now says which models it means.

    WHAT THIS SPLIT IS NOT
    It is i.i.d. over ROWS, not over survey clusters. NHANES sampled people
    inside 49 primary sampling units, and because the split ignores that, every
    PSU in the test block also appears in the training block -- all 49 of them.
    So the test rows come from neighbourhoods the model has already seen, and the
    absolute accuracies here carry whatever optimism that buys.

    That is the honest statement, and it is the reason the paper's claims are
    PAIRED DIFFERENCES between rungs on the identical test rows rather than
    absolute accuracies. Any cluster optimism is common to both arms of a
    contrast and largely cancels. That is measured, not asserted:
    ablation.cluster_holdout() refits every rung on two folds that share no PSU
    with their own test side and recomputes all six contrasts there, and
    ablation.json's cluster_holdout block reports the absolute optimism next to
    the contrast-by-contrast difference. The
    design-based intervals wrapped around these estimates are cluster-robust for
    the variance, which is a separate matter from the point estimate's optimism.
    """
    y = df["Severity4"].to_numpy()
    idx = np.arange(len(df))
    rest, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED,
                                stratify=y)
    tr, va = train_test_split(rest, test_size=VAL_SIZE, random_state=SEED,
                              stratify=y[rest])
    return np.sort(tr), np.sort(va), np.sort(te)


# ------------------------------------------------------------ survey design --
def design_arrays(df, idx=None):
    """(stratum, psu, weight) for all rows or a subset. Primary cycle only."""
    d = df if idx is None else df.iloc[idx]
    return (d[STRATUM].to_numpy(), d[PSU].to_numpy(), d[MECWT].to_numpy())


def psu_bootstrap(strata, psu, rng):
    """
    One NAIVE cluster-bootstrap resample: draw n_h PSUs with replacement in each
    stratum and take all their rows.

    Kept because it is the obvious thing to do and it is instructive that it is
    wrong here. This design has 2 PSUs in 23 of its 24 strata and 3 in the
    remaining one, and drawing n_h with replacement gives a variance whose
    expectation is (n_h - 1)/n_h of the truth -- half, at the n_h = 2 that
    describes almost every stratum here -- so this UNDERSTATES the variance
    rather than correcting it. Use rao_wu_weights for anything that goes in the
    paper.

    Returns row positions (into the arrays passed in), with multiplicity.
    """
    rows = np.arange(len(strata))
    out = []
    for h in np.unique(strata):
        in_h = rows[strata == h]
        clusters = np.unique(psu[in_h])
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        for c in picked:
            out.append(in_h[psu[in_h] == c])
    return np.concatenate(out) if out else rows


def design_df(strata, psu):
    """Degrees of freedom for this design: PSUs minus strata."""
    n_psu = len({(h, p) for h, p in zip(strata, psu)})
    return int(n_psu - len(set(strata)))


def t_and_p(obs, se, df):
    """
    Two-sided t test of obs = 0 with a bootstrap SE, degenerate cases included.

    Returns {"t", "p_value", "degenerate"}. It exists because three scripts each
    wrote this by hand and two of them wrote it differently: a zero replicate
    spread was p = 1 in benchmark.py and ablation.py and p = 0 in weighted.py, so
    the same degenerate contrast would have been reported as no evidence or as
    total evidence depending on which file computed it. One function, one answer.

    Five cases:

      not finite                a NaN obs or a NaN se is not a degenerate test,
                                it is an absent one. This branch used to be
                                missing, and the consequence was specific and
                                bad: `if se > 0` is False for NaN, `obs == 0` is
                                False for NaN, so a contrast whose bootstrap SE
                                was undefined fell through to the last case and
                                was published as p = 0.0 - the study's strongest
                                result - with a `degenerate` string asserting a
                                zero replicate spread, which is not what had
                                happened. p comes back 1.0, the conservative
                                answer, and the reason says which input was NaN.
      df <= 0                   no residual degrees of freedom, so stats.t.sf
                                returns NaN and the "p_value is always a float"
                                promise below was false. design_df returns 0 for
                                a single-stratum subset, which is exactly the
                                situation this guard was written for.
      se > 0                    the ordinary test.
      se = 0 and obs = 0        the two models made identical predictions on
                                every row, so there is nothing to test and p = 1.
                                0/0 is not infinite evidence.
      se = 0 and obs != 0       a real difference that no replicate moved. The
                                limit of the t statistic is infinite, so the
                                limit of p is 0 - but "p = 0" is a statement the
                                bootstrap cannot support, and `float("inf")` is
                                not JSON either: every script here writes with
                                allow_nan=False, so an infinite t would have
                                crashed the dump at the end of a long run rather
                                than at the point of the problem. `t` comes back
                                None and `degenerate` says why, so a reader sees
                                a missing statistic instead of a fabricated one.

    p_value is always a finite float, because callers Holm-correct it and compare
    it to 0.05; only `t` can be None. On this cohort no contrast in any of the
    three scripts reaches the last case - it is a guard, and the two-way split
    above is what it is really for.
    """
    obs, se = float(obs), float(se)
    if not math.isfinite(obs) or not math.isfinite(se):
        which = ("delta" if not math.isfinite(obs) else "standard error")
        return {"t": None, "p_value": 1.0,
                "degenerate": (f"the {which} is not finite, so there is no test "
                               f"here: delta {obs}, se {se}. Reported as p = 1 "
                               f"because an undefined statistic is no evidence, "
                               f"not overwhelming evidence")}
    if not df > 0:
        return {"t": None, "p_value": 1.0,
                "degenerate": (f"{df} residual degrees of freedom, so the t "
                               f"distribution is undefined; delta {obs:+.6f}, "
                               f"se {se:.6g}")}
    if se > 0:
        t = obs / se
        return {"t": float(t),
                "p_value": float(2 * stats.t.sf(abs(t), df)),
                "degenerate": None}
    if obs == 0:
        return {"t": 0.0, "p_value": 1.0,
                "degenerate": "identical predictions: no difference to test"}
    return {"t": None, "p_value": 0.0,
            "degenerate": (f"delta {obs:+.6f} with zero replicate spread; the t "
                           f"statistic is unbounded and p = 0 is its limit, not "
                           f"a measurement")}


def holm(pvals):
    """
    Holm step-down adjusted p-values, in the order the p-values came in.

    Here for the same reason t_and_p is here: this was written by hand in
    ablation.py and again in benchmark.py, and weighted.py did not have a copy at
    all - which is why weighted.py's three contrasts went out uncorrected while
    both of its neighbours corrected theirs. The two hand-written copies were
    checked against each other over 2000 random families, ties included, and
    agreed to 0.0e+00, so no number that has already been published moves when
    they are replaced by this one. That is luck rather than design: two
    implementations of a step-down procedure that differ only in how they break
    ties give different answers on exactly the families where ties occur, and
    paired accuracy contrasts on one test block tie often.

    Step-down, not step-up: sort ascending, multiply the k-th smallest by
    (m - k), and take a running maximum so the sequence cannot decrease. The
    running maximum is the part that is easy to leave out and hard to notice,
    because without it the adjusted values are still mostly monotone.

    Order is preserved on the way out. Callers zip these straight back onto their
    contrast list, so returning them sorted would silently attach each adjusted
    p-value to the wrong contrast.
    """
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return [float(v) for v in adj]


def rao_wu_weights(strata, psu, rng, base):
    """
    One replicate of the Rao-Wu rescaled bootstrap: returns replicate WEIGHTS,
    not a row selection.

    This is the resampling scheme that is actually valid for a design with two
    PSUs per stratum, and it is the one the paper uses. For stratum h with n_h
    PSUs, draw m_h = n_h - 1 of them with replacement, count the draws r_hi, and
    set

        w*_hi = w_hi * (n_h / m_h) * r_hi

    At n_h = 2 -- 23 of the 24 strata here -- that doubles one PSU and zeroes the
    other, which is the same
    half-sample logic as balanced repeated replication. The resulting variance
    estimator has the right expectation, and the degrees of freedom come out at
    (number of PSUs - number of strata), matching the linearised estimator in
    weighted.py rather than pretending there are thousands of independent rows.

    Every statistic then has to accept sample weights -- which sklearn's metrics
    do -- instead of being recomputed on a duplicated row list.

    `base` has no default on purpose. Pass the survey weight to get intervals
    for a POPULATION quantity; pass None to get weights of one, which gives
    cluster-and-stratum-robust intervals for a SAMPLE quantity. Both are
    legitimate and they answer different questions, so the caller has to say
    which one it means. ablation.py and benchmark.py pass None deliberately:
    their estimand is a paired difference between two models on the same sample,
    not a population total. weighted.py passes the weight.
    """
    w = np.ones(len(strata), dtype=float) if base is None else np.asarray(
        base, dtype=float).copy()
    out = np.zeros_like(w)
    for h in np.unique(strata):
        in_h = np.flatnonzero(strata == h)
        clusters = np.unique(psu[in_h])
        n_h = len(clusters)
        if n_h < 2:                      # nothing to resample; keep as is
            out[in_h] = w[in_h]
            continue
        m_h = n_h - 1
        draws = rng.choice(n_h, size=m_h, replace=True)
        r = np.bincount(draws, minlength=n_h)
        for j, c in enumerate(clusters):
            rows_c = in_h[psu[in_h] == c]
            out[rows_c] = w[rows_c] * (n_h / m_h) * r[j]
    return out


def describe(df, att):
    """Print everything the paper needs for its cohort and label paragraphs."""
    w = df[MECWT].to_numpy()
    n = len(df)

    print(f"COHORT  ({att['cycle']}, MEC weight {att['weight_column']})")
    print(f"  CBC records                      {att['cbc_records']:,}")
    print(f"  after merge with DEMO            {att['after_demo_merge']:,}")
    print(f"  complete on all CBC features     {att['complete_on_all_features']:,}")
    print(f"  with a WHO threshold assigned    {att['with_who_threshold']:,}")
    print(f"  excluded                         {att['excluded_total']:,} "
          f"({att['excluded_pct']}%)")

    ap = att["attrition_profile"]
    if ap["n_dropped"]:
        d, k = ap["dropped"], ap["retained"]
        print("\nWHO WAS EXCLUDED  (complete-case analysis is only safe if these match)")
        print(f"  {'':<22}{'dropped':>10}{'retained':>10}")
        for key, lab in [("n", "n"), ("mean_age", "mean age, y"),
                         ("sd_age", "SD age, y"),
                         ("pct_under_2", "% under 2"),
                         ("pct_under_18", "% under 18"),
                         ("pct_65_plus", "% 65+"), ("pct_female", "% female")]:
            print(f"  {lab:<22}{d[key]:>10,}{k[key]:>10,}")
        smd = ap["age_standardised_difference"]
        print(f"  age gap                {ap['age_gap_years']:>10} y"
              f"   standardised {smd}  ({ap['smd_convention']})")
        print(f"  MCAR contradicted on age?  "
              f"{'YES' if ap['mcar_contradicted_on_age'] else 'not by age'}"
              f"   - {ap['mcar_note']}")
        print(f"  {ap['note']}")

    ig = att["integrity"]
    print("\nRAW-VALUE INTEGRITY  (screened, not cleaned)")
    print(f"  values screened                  {ig['n_values_screened']:,}")
    print(f"  outside containment bounds       "
          f"{ig['values_outside_containment_bounds']:,}")
    print(f"  non-positive / non-finite        {ig['non_positive_values']:,}"
          f" / {ig['non_finite_values']:,}")
    print(f"  duplicate SEQN                   {ig['duplicate_seqn']:,}")
    print(f"  rows dropped by this screen      "
          f"{ig['rows_dropped_by_this_screen']:,}")
    if ig["values_outside_containment_bounds"]:
        for col, c in ig["by_column"].items():
            if c["outside_bounds"]:
                print(f"    {col}: {c['outside_bounds']} outside "
                      f"{c['containment_bounds']}, observed range "
                      f"[{c['min']}, {c['max']}]")
    print(f"  {ig['policy']}")

    pg = att["pregnancy"]
    print("\nPREGNANCY, AND WHERE IT IS A GUESS")
    print(f"  recorded pregnant                {pg['recorded_pregnant']:,}")
    print(f"  recorded not pregnant            {pg['recorded_not_pregnant']:,}")
    print(f"  'could not ascertain' (code 3)   {pg['could_not_ascertain_code3']:,}")
    print(f"  women 12-49                      {pg['women_12_49']:,}")
    print(f"  of those, never asked            {pg['women_12_49_not_ascertained']:,}")
    print(f"  positives that could flip off    {pg['labels_that_could_flip_off']:,}"
          f" ({pg['flip_pct_of_all_positives']}% of all positives)")
    print(f"  so: {pg['direction']}")

    print("\nCOMPOSITION")
    print(f"  men                              {int((df[SEX] == 1).sum()):,}")
    print(f"  women                            {int((df[SEX] == 2).sum()):,}")
    print(f"  children < 18 y                  {int((df[AGE] < 18).sum()):,}")
    print(f"  age range                        "
          f"{df[AGE].min():.0f}-{df[AGE].max():.0f} y")

    print("\nWHO THRESHOLD APPLIED (g/dL)")
    wt_aud = who_threshold_audit(df)
    for t, c in wt_aud["rows_per_threshold"].items():
        print(f"  {t:>5}                            {c:,}")
    wrong_cut = wt_aud["rows_whose_cutoff_is_not_12"]
    print(f"  not 12.0, i.e. people the manuscript's single cut is wrong for"
          f"   {wrong_cut:,} ({wt_aud['pct_whose_cutoff_is_not_12']:.2f}%)")

    print("\nLABEL: WHO vs the manuscript's sex-blind Hb<12")
    a_who, a_old = df["Anemia"], df["Anemia_hb12"]
    print(f"  WHO age/sex/pregnancy specific   {a_who.sum():,} anemic "
          f"({a_who.mean()*100:.2f}%)")
    print(f"  single 12.0 g/dL cut             {a_old.sum():,} anemic "
          f"({a_old.mean()*100:.2f}%)")
    flip = (a_who != a_old)
    print(f"  labels that change               {flip.sum():,} ({flip.mean()*100:.2f}%)")
    print(f"    missed by the 12.0 cut         {int(((a_old==0)&(a_who==1)).sum()):,}")
    print(f"    falsely flagged by it          {int(((a_old==1)&(a_who==0)).sum()):,}")

    print("\nSEVERITY (WHO four-class, then merged as in the manuscript)")
    for i, name in enumerate(SEVERITY_NAMES_4):
        c = int((df.Severity4 == i).sum())
        print(f"  {name:<16s}                 {c:,} ({c/n*100:.2f}%)")
    counts3 = df["Severity"].value_counts().sort_index()
    ratio = counts3.max() / max(counts3.min(), 1)
    print(f"  merged 3-class imbalance ratio   {ratio:.1f}:1")
    print(f"  Moderate and Severe are merged because Severe has "
          f"{int((df.Severity4 == 3).sum())} members in the whole cohort")

    print("\nBASELINES the manuscript never reported")
    print(f"  binary no-information rate       "
          f"{max(a_who.mean(), 1-a_who.mean())*100:.2f}%")
    print(f"  severity no-information rate     {counts3.max()/n*100:.2f}%")
    print(f"  PR-AUC no-skill reference        {a_who.mean():.4f}  "
          f"(prevalence, not the majority rate)")

    print("\nSURVEY DESIGN (ignored by all 20 reviewed papers)")
    print(f"  strata ({STRATUM})                {df[STRATUM].nunique()}")
    _per_h = (df.groupby(STRATUM, observed=True)[PSU].nunique()
                .value_counts().sort_index())
    print(f"  PSU labels used ({PSU})        "
          f"{sorted(int(v) for v in df[PSU].unique())}   "
          f"(labels, not counts -- they repeat across strata)")
    print("  PSUs per stratum                 "
          + ",  ".join(f"{int(c)} stratum has {int(k)}" if c == 1
                       else f"{int(c)} strata have {int(k)}"
                       for k, c in _per_h.items()))
    print(f"  distinct stratum x PSU cells     "
          f"{df.groupby([STRATUM, PSU], observed=True).ngroups}")
    print(f"  design degrees of freedom        "
          f"{design_df(df[STRATUM].to_numpy(), df[PSU].to_numpy())}"
          f"   (PSU cells minus strata)")
    print(f"  population represented           {w.sum()/1e6:.1f} million")
    print(f"  unweighted prevalence            {a_who.mean()*100:.2f}%")
    print(f"  weighted prevalence              "
          f"{np.average(a_who, weights=w)*100:.2f}%")

    tr, va, te = split3(df)
    print("\nSHARED SPLIT  (stratified on Severity4)")
    for name, part in [("train", tr), ("validation", va), ("test", te)]:
        y = df["Anemia"].to_numpy()[part]
        s = df["Severity4"].to_numpy()[part]
        print(f"  {name:<12} {len(part):>7,}   anemic {int(y.sum()):>4,} "
              f"({y.mean()*100:5.2f}%)   severe {int((s == 3).sum())}")
    print("  validation exists so estimator and threshold choice never touch test")

    print("\nLEAKAGE LADDER")
    seen = set()
    for rung, cols in LADDER.items():
        have = set(cols)
        if HGB in have and set(DEMO_FEATURES) <= have:
            tag = "EXACT"          # label is a deterministic function of input
        elif HGB in have:
            tag = "LABEL"
        elif any(route <= have for route in HGB_ROUTES):
            tag = "IDENT"          # an analyser identity rebuilds hemoglobin
        elif {MCV, RBC} <= have:
            tag = "hct"            # rebuilds hematocrit, not hemoglobin
        else:
            tag = "-"
        seen.add(tag)
        demo = "+demo" if set(DEMO_FEATURES) <= have else ""
        print(f"  {rung:<24s} {tag:<5s} {demo:<5s} "
              f"{[c for c in cols if c not in DEMO_FEATURES]}")
    # Only legend what the table above actually shows. Every rung of the current
    # ladder keeps both MCV and RBC, so the "-" tag never fires, and printing its
    # legend anyway described a row that cannot exist. The branch stays, because
    # the tagger has to classify a rung that drops one of them; the legend is
    # conditional, because a reader should not have to check.
    legend = [("EXACT", "EXACT = label fully determined"),
              ("LABEL", "LABEL = hemoglobin given directly"),
              ("IDENT", f"IDENT = one of {len(HGB_ROUTES)} analyser identities "
                        "reconstructs hemoglobin"),
              ("hct", "hct   = {MCV, RBC} reconstructs hematocrit exactly"),
              ("-", "-     = no exact route. NOT the same as uninformative: "
                    "see LADDER")]
    for tag, line in legend:
        if tag in seen:
            print(f"  {line}")


def main():
    df, att = load_cycle(PRIMARY, verbose=True)
    print("\nREPLICATION CYCLES (unweighted; designs are not poolable)")
    reps = load_replication(verbose=True)
    print(f"  primary   n = {len(df):,}  anemia {df.Anemia.mean()*100:.2f}%")
    total = len(df) + sum(len(d) for d, _ in reps.values())
    print(f"  {len(reps) + 1} cycles, {total:,} records in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
