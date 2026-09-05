"""
Step 4 - the comparator benchmark the manuscript never reported.

The submitted paper gives one number for one model with no baseline, so a reader
cannot tell whether 99% is skill or arithmetic. This script puts nine model entries
- two of them trivial baselines - on the same cohort, the same folds and the same
rows, at three rungs of the leakage ladder.

What it answers
  R4-2  compare against a baseline and against other models
  R4-5  report variance, confidence intervals, significance testing
  R6-2  why Random Forest?
  R6-4  how does this compare with existing approaches?

Six things are deliberate, and each one is a fix to how the first version of
this script did it.

  The estimator is chosen on a VALIDATION split, never on the test set. The
  selection rule is declared in SELECT_METRIC before any number is looked at,
  and it is validation PR-AUC rather than accuracy, because at 9.6% prevalence
  accuracy is dominated by the negative class. The winner and Random Forest are
  BOTH reported: the winner because that is the honest answer to "which model is
  best", and Random Forest because the manuscript used one, so the leakage
  contrast has to be like-for-like against it.

  Every estimator gets balanced class handling. XGBoost has no class_weight, so
  BalancedXGB computes balanced per-sample weights inside fit; without that the
  nine-way table would partly be comparing weighting schemes rather than
  algorithms. HistGradientBoosting gets class_weight="balanced" for the same
  reason - the first version left both unbalanced.

  Confidence intervals and p-values are design-based, from Rao-Wu rescaled
  bootstrap replicate weights on (PSUs - strata) degrees of freedom. The i.i.d.
  row bootstrap is computed too, and their RATIO is the column printed next to
  every contrast - "SE x row", design over row - with both standard errors
  themselves in benchmark.json under test_accuracy_design_se and
  test_accuracy_row_se. The ratio is the printed one because it is the number the
  claim is about: NHANES is a clustered sample, and asserting that in a methods
  sentence is cheaper than showing what it does to an interval.

  The significance comparisons are a pre-declared family of six, Holm-corrected.
  McNemar is reported next to the design-based test because McNemar is what the
  literature uses and it assumes independent rows.

  Thresholds are tuned on validation and applied unchanged to test. Sensitivity
  at fixed specificity is reported, because a screening tool is used at an
  operating point, not at 0.5.

  Cross-validation is repeated: 5 folds x 4 repeats = 20 fits, so the fold
  spread is a real estimate of seed sensitivity rather than one lucky partition.

  StandardScaler appears here, and only here, wrapped around the two estimators
  that need it. The trees are scale-invariant, which is why the manuscript's
  scaler was a no-op.

Outputs
  results/benchmark.json
  stdout   the comparator tables for the paper

Run:  python src/benchmark.py
"""

import json
import sys
import time

import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

import dataprep as dp
from paths import RESULTS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = dp.SEED
N_FOLDS = dp.N_FOLDS       # one place, dataprep; see the note there
N_REPEATS = 4                  # 5 x 4 = 20 fits per estimator per rung
N_REP = dp.N_REP               # one place, dataprep; see the note there
N_BOOT = dp.N_REP              # matching i.i.d. row bootstrap replicates: the
                               # point of the comparison is that the two schemes
                               # differ in what they resample and in nothing
                               # else, so they must not differ in replicate
                               # count either. Both used to be 500 and moved
                               # together only by coincidence.
OUT_PATH = RESULTS / "benchmark.json"

# Declared before any result was inspected. PR-AUC on the VALIDATION split,
# because accuracy at 9.6% prevalence is mostly a statement about the negative
# class, and because selecting on test is the mistake this paper is about.
SELECT_METRIC = "val_pr_auc"
LIKE_FOR_LIKE = "rf"           # the manuscript's estimator, kept as comparator
FIXED_SPECIFICITY = 0.90       # operating point for the sensitivity report

# the three rungs worth comparing models on:
#   L1 - what the manuscript actually submitted
#   L4 - exact-identity-free CBC only
#   L6 - identity-free CBC plus the demographics a clinician legitimately knows
RUNGS = ["L1_paper_leaky", "L4_identity_free", "L6_identity_free_demo"]
HEADLINE = "L6_identity_free_demo"

# Which entries of models() are not estimators. Named once because three places
# needed the distinction - best_real_model(), the stdout table, and fig1's
# "N of them trivial baselines" caption - and all three had the pair written out
# as a literal tuple, so adding a third baseline would have been correct in two
# places and wrong in the third with nothing to catch it.
BASELINES = ("majority", "stratified")

class BalancedXGB(XGBClassifier):
    """
    XGBoost with class_weight="balanced" semantics.

    XGBoost has no class_weight; it has scale_pos_weight, which is binary-only.
    Every other estimator in this comparison is balanced, so leaving XGBoost
    unbalanced on the three-class severity problem would turn part of the
    nine-way table into a comparison of weighting schemes. Computing the weights
    inside fit also keeps the estimator usable inside cross_val_score, which
    cannot route a sample_weight without metadata routing enabled globally.
    """

    def fit(self, X, y, **kw):
        kw.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kw)


def models():
    """
    Nine model entries, built fresh on every call so no state leaks across rungs.

    "majority" and "stratified" are the baselines the manuscript omitted: any
    honest accuracy claim has to be read against them.

    The same nine serve the binary and the severity target. There is no
    multiclass switch, because none of these nine needs different construction
    for three classes: sklearn's own multiclass handling covers them, and
    xgboost is configured for it in BalancedXGB. An earlier version took a
    `multiclass` argument and ignored it, which promised a distinction that did
    not exist. evaluate() does take one, and that one is live: it changes which
    metrics are computed.
    """
    return {
        # -- baselines ------------------------------------------------------
        "majority": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=SEED),
        # -- scale-sensitive: StandardScaler is genuinely required ----------
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEED)),
        # SVC(probability=True) is deprecated in sklearn 1.9; the sigmoid
        # calibration it used to do internally is now explicit.
        "svm_rbf": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", cache_size=1000,
                    random_state=SEED), ensemble=False, cv=3)),
        # -- trees: scale-invariant, no scaler ------------------------------
        "dtree": DecisionTreeClassifier(class_weight="balanced",
                                        random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=dp.N_TREES,
                                     class_weight="balanced",
                                     random_state=SEED, n_jobs=1),
        "extratrees": ExtraTreesClassifier(n_estimators=dp.N_TREES,
                                           class_weight="balanced",
                                           random_state=SEED, n_jobs=1),
        "histgb": HistGradientBoostingClassifier(class_weight="balanced",
                                                 random_state=SEED),
        "xgboost": BalancedXGB(n_estimators=dp.N_TREES, max_depth=6,
                               learning_rate=0.1, tree_method="hist",
                               eval_metric="logloss", random_state=SEED,
                               n_jobs=-1),
    }


_REP_CACHE = {}


def replicate_weights(strata, psu, n=N_REP, seed=SEED):
    """
    n Rao-Wu replicate weight vectors for this design, built once and reused.

    base=None: the replicate weights start from one, not from MECWT. Everything
    in this file is therefore an interval for a SAMPLE quantity that is robust
    to the clustering and stratification, not an estimate of a population
    quantity. The population estimates live in weighted.py, which passes the
    survey weight.

    Reading a CI from benchmark.json as a population interval would be wrong, and
    weighted.py measures by how much: for the same L6 feature set its out-of-fold
    accuracy is 0.9630 unweighted and 0.9733 weighted, a gap of 0.0103 against a
    design-based 95% interval whose half-width is 0.0054. So the gap is roughly
    twice the margin of error - the unweighted figure lands outside the population
    interval - though smaller than the full interval width, which is what this
    note used to claim and is a different and false comparison.

    "For the same feature set", not "on the same rows", which is what this used to
    say. weighted.py's figures are out-of-fold over all 12,156 rows and every
    number in this file is on the 2,432-row test block, so the two are not
    computed on one set of rows at all. The comparison is still the right one to
    make - it is unweighted against weighted for one model specification - but it
    is not paired, and calling it paired overstated it.
    """
    key = (strata.tobytes(), psu.tobytes(), n, seed)
    if key not in _REP_CACHE:
        rng = np.random.default_rng(seed)
        _REP_CACHE[key] = np.stack(
            [dp.rao_wu_weights(strata, psu, rng, base=None) for _ in range(n)])
    return _REP_CACHE[key]


_ROW_CACHE = {}


def row_indices(k, n=N_BOOT, seed=SEED):
    """n i.i.d. row-bootstrap index sets: the variance the literature reports."""
    key = (k, n, seed)
    if key not in _ROW_CACHE:
        rng = np.random.default_rng(seed)
        _ROW_CACHE[key] = rng.integers(0, k, size=(n, k))
    return _ROW_CACHE[key]


def design_ci(stat_fn, strata, psu, alpha=0.05, n=N_REP):
    """
    Design-based interval for any statistic that accepts a sample_weight.

    stat_fn(w) must return the statistic under weights w, and stat_fn(None) the
    point estimate. Variance comes from the spread of the replicates about the
    point estimate, and the interval is a t interval on (PSUs - strata) degrees
    of freedom - 25 here, not n - 1 = 2,431.
    """
    theta = float(stat_fn(None))
    W = replicate_weights(strata, psu, n)
    vals = np.array([stat_fn(w) for w in W], dtype=float)
    vals = vals[np.isfinite(vals)]      # a replicate can zero out a rare class
    se = float(np.sqrt(np.mean((vals - theta) ** 2)))
    df = dp.design_df(strata, psu)
    t = stats.t.ppf(1 - alpha / 2, df)
    # A zero replicate spread collapses the interval to a point, and a point
    # printed as "[0.000, 0.000]" reads as a measurement of certainty rather than
    # an absence of one. The majority baseline's recall does exactly this - it
    # predicts the negative class for everyone, so recall is 0 under every set of
    # replicate weights - and so does any estimator that is right on every test
    # row. The interval is returned unchanged; `degenerate` is what says not to
    # read it as one.
    degenerate = None
    if se == 0.0:
        degenerate = (f"all {len(vals)} replicates returned {theta:.6g}: zero "
                      f"replicate spread, so this interval is a point and not an "
                      f"uncertainty. Expected when the statistic sits on a "
                      f"boundary the design cannot move it off.")
    return {"value": theta, "se": se, "df": df,
            "ci95": [theta - t * se, theta + t * se],
            "degenerate": degenerate}


def design_test(stat_fn, strata, psu, n=N_REP):
    """
    design_ci plus a two-sided t test of stat = 0, for paired differences.

    The degenerate cases - a zero replicate spread with, and without, a nonzero
    estimate - belong to dataprep.t_and_p, which is also what ablation.py and
    weighted.py call, so all three files answer them the same way. They did not
    before: weighted.py returned p = 0 where this file returned p = 1.
    """
    r = design_ci(stat_fn, strata, psu, n=n)
    r.update(dp.t_and_p(r["value"], r["se"], r["df"]))
    return r


def row_se(stat_fn, k, n=N_BOOT):
    """
    The same statistic's standard error under i.i.d. row resampling.

    Returns the SE alone. It used to return (se, theta) and both call sites
    discarded theta into `_`, which is worse than it sounds: theta here is
    stat_fn(None), the point estimate, and having it available from the SE
    function invites a caller to take the point estimate from the bootstrap
    helper instead of from the model evaluation, which is how two tables end up
    quoting the same statistic to different precision.

    ddof=1 on purpose, and it does NOT match design_ci's ddof=0 above. The two
    are different estimators of different things: this one is the spread of a
    resample around its own mean, the design version is the spread of replicates
    around the fixed sample estimate, which is what the Rao-Wu scheme defines.
    The ratio of the two appears in the JSON and the discrepancy the convention
    introduces is 1/(2n) = 0.1% at n = 500, well below the third decimal anything
    is reported to. Written down because "why is one ddof=1 and the other 0" is
    the question, and the answer is that they are not the same formula.
    """
    vals = [stat_fn(idx) for idx in row_indices(k, n)]
    return float(np.std(vals, ddof=1))


def holm(pvals):
    """
    Delegates to dp.holm. See its docstring for why there is only one copy now.

    This file and ablation.py each had their own hand-written version, spelled
    differently and behaving identically; weighted.py had neither, which is the
    part that actually cost something.
    """
    return dp.holm(pvals)


def mcnemar_test(y, pred_a, pred_b):
    """
    Paired McNemar, reported because it is what this literature uses.

    n10 = a right / b wrong, n01 = a wrong / b right. Exact binomial when the
    discordant count is small, chi-square with continuity correction otherwise.
    It assumes the rows are independent, which in a clustered survey they are
    not, so the design-based test next to it is the one to read.
    """
    ca, cb = (pred_a == y), (pred_b == y)
    n11, n10 = int((ca & cb).sum()), int((ca & ~cb).sum())
    n01, n00 = int((~ca & cb).sum()), int((~ca & ~cb).sum())
    r = mcnemar([[n11, n10], [n01, n00]],
                exact=(n10 + n01) < 25, correction=True)
    return {"a_right_b_wrong": n10, "a_wrong_b_right": n01,
            "statistic": float(r.statistic), "p_value": float(r.pvalue),
            "assumes": "independent rows; NHANES rows are clustered in PSUs"}


def calibration(y, proba, bins=10):
    """Brier score plus a quantile-binned reliability curve for figures.py."""
    out = {"brier": float(brier_score_loss(y, proba))}
    try:
        # a constant-output baseline has no usable reliability curve
        frac, mean_pred = calibration_curve(y, proba, n_bins=bins,
                                            strategy="quantile")
        out["bin_observed"] = [float(v) for v in frac]
        out["bin_predicted"] = [float(v) for v in mean_pred]
    except (ValueError, IndexError):
        out["bin_observed"] = None
        out["bin_predicted"] = None
    return out


def pick_thresholds(yva, pva, spec=FIXED_SPECIFICITY):
    """
    Choose operating points on the VALIDATION split only.

    Three of them, because 0.5 is an arbitrary default that no screening
    programme would use:
      f1        the threshold maximising F1
      youden    the threshold maximising sensitivity + specificity - 1
      spec90    the highest sensitivity available at >= 90% specificity

    Tuning these on validation and then applying them unchanged to test is the
    whole point; tuning on test is a smaller version of the leak this paper is
    about.

    sklearn's roc_curve prepends an infinite threshold, the point that predicts
    everything negative. For a constant-score baseline that point can win the
    argmax, and an infinity in a JSON file is both meaningless as an operating
    point and invalid JSON. Those are reported as null with a reason instead.
    """
    def finite(t, why):
        """None, not inf: 'no usable threshold' is a result, not a number."""
        t = float(t)
        return {"threshold": None, "unavailable": why} if not np.isfinite(t) \
            else {"threshold": t}

    grid = np.unique(np.round(pva, 4))
    grid = grid[(grid > 0) & (grid < 1)]
    # An empty grid means every validation score is exactly 0 or exactly 1, which
    # is what a constant-score baseline produces. Falling back to [0.5] keeps the
    # code running, but the row it then writes says "maximises validation F1"
    # about a single candidate that maximised nothing. The fallback is kept - a
    # crash here would take down the whole stage over a Dummy estimator - and the
    # rule string now says what actually happened.
    degenerate_grid = len(grid) == 0
    if degenerate_grid:
        grid = np.array([0.5])
    f1s = [f1_score(yva, (pva >= t).astype(int), zero_division=0) for t in grid]
    fpr, tpr, thr = roc_curve(yva, pva)
    j = tpr - fpr
    ok = (1 - fpr) >= spec
    no_thr = "roc_curve returned only its infinite endpoint: the scores are " \
             "constant, so no threshold separates anything"
    f1_rule = ("no interior validation score exists (every score is 0 or 1), so "
               "0.5 is the only candidate and nothing was maximised"
               if degenerate_grid else "maximises validation F1")
    return {
        "default": {"threshold": 0.5, "rule": "the arbitrary default"},
        "f1": {"threshold": float(grid[int(np.argmax(f1s))]),
               "rule": f1_rule,
               "grid_degenerate": degenerate_grid,
               "val_f1": float(max(f1s))},
        "youden": {**finite(thr[int(np.argmax(j))], no_thr),
                   "rule": "maximises validation sensitivity + specificity - 1",
                   "val_youden_j": float(j.max())},
        f"spec{int(spec*100)}": {
            **(finite(thr[ok][int(np.argmax(tpr[ok]))], no_thr) if ok.any()
               else {"threshold": None,
                     "unavailable": f"no validation threshold reaches "
                                    f"{spec:.0%} specificity"}),
            "rule": f"highest validation sensitivity at >= {spec:.0%} specificity",
            "val_sensitivity": float(tpr[ok].max()) if ok.any() else 0.0},
    }


def evaluate(est, X, y, tr, va, te, strata, psu, multiclass=False):
    """
    Score one estimator on one rung.

    Returns (json_dict, test_pred, test_proba, fit_seconds).

    Fitted on train only. Validation carries the selection metric and the
    threshold search; test carries everything reported. Predictions come back so
    main() can run paired tests on them; they are too large for the JSON.

    fit_seconds is returned OUTSIDE the json_dict on purpose. It used to be a
    field inside it, which meant benchmark.json's hash changed on every run for
    reasons that had nothing to do with any result - and a manifest that always
    differs cannot show that a change was inert. It is printed, never stored.
    """
    t0 = time.time()
    cv = cross_val_score(
        est, X[tr], y[tr], scoring="accuracy",
        cv=RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=SEED),
        n_jobs=-1)
    est.fit(X[tr], y[tr])
    pred, yte = est.predict(X[te]), y[te]
    pred_va, yva = est.predict(X[va]), y[va]

    res = {
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_sd": float(cv.std(ddof=1)),
        "cv_accuracy_min": float(cv.min()),
        "cv_accuracy_max": float(cv.max()),
        "cv_n_fits": int(len(cv)),
        "val_accuracy": float(accuracy_score(yva, pred_va)),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }
    secs = round(time.time() - t0, 1)

    acc = design_ci(lambda w: accuracy_score(yte, pred, sample_weight=w),
                    strata, psu)
    res["test_accuracy_ci95"] = acc["ci95"]
    res["test_accuracy_design_se"] = acc["se"]
    rse = row_se(lambda i: accuracy_score(yte, pred) if i is None
                 else accuracy_score(yte[i], pred[i]), len(yte))
    res["test_accuracy_row_se"] = rse
    res["se_ratio_design_over_row"] = round(acc["se"] / max(rse, 1e-12), 2)

    proba = est.predict_proba(X[te])
    if multiclass:
        res["macro_f1"] = float(f1_score(yte, pred, average="macro"))
        res["per_class_recall"] = [
            float(v) for v in recall_score(yte, pred, average=None,
                                           labels=sorted(set(y)),
                                           zero_division=0)]
        try:
            res["roc_auc_ovr_macro"] = float(
                roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
        except ValueError:
            res["roc_auc_ovr_macro"] = None
        res["val_macro_f1"] = float(f1_score(yva, pred_va, average="macro"))
        return res, pred, None, secs

    p1, pva = proba[:, 1], est.predict_proba(X[va])[:, 1]
    res["roc_auc"] = float(roc_auc_score(yte, p1))
    res["pr_auc"] = float(average_precision_score(yte, p1))
    res["pr_auc_no_skill"] = float(yte.mean())
    res["val_pr_auc"] = float(average_precision_score(yva, pva))
    res["val_roc_auc"] = float(roc_auc_score(yva, pva))
    res["precision"] = float(precision_score(yte, pred, zero_division=0))
    res["recall"] = float(recall_score(yte, pred, zero_division=0))
    res["f1"] = float(f1_score(yte, pred, zero_division=0))
    res["calibration"] = calibration(yte, p1)

    for key, fn in [("recall", lambda w: recall_score(
                        yte, pred, sample_weight=w, zero_division=0)),
                    ("precision", lambda w: precision_score(
                        yte, pred, sample_weight=w, zero_division=0)),
                    ("pr_auc", lambda w: average_precision_score(
                        yte, p1, sample_weight=w))]:
        ci = design_ci(fn, strata, psu)
        res[f"{key}_ci95"] = ci["ci95"]
        # The majority baseline predicts "not anemic" for everyone, so its recall
        # is 0 under every replicate weighting and this interval came out as the
        # flat [0.0, 0.0]. Carrying the flag is what stops that being read as a
        # tight interval.
        if ci["degenerate"]:
            res[f"{key}_ci95_degenerate"] = ci["degenerate"]

    # operating points chosen on validation, applied unchanged to test
    res["thresholds"] = pick_thresholds(yva, pva)
    for tag, spec in res["thresholds"].items():
        t = spec["threshold"]
        if t is None:              # no usable operating point; see pick_thresholds
            continue
        # >= t, the same convention as the threshold search. At t = 0.5 this can
        # disagree with est.predict() for a tie at exactly 0.5, which argmax
        # sends to class 0; res["test_accuracy"] above is the predict() version
        # and is the headline. Both are reported so the gap is visible.
        hard = (p1 >= t).astype(int)
        tn = int(((hard == 0) & (yte == 0)).sum())
        spec["test_sensitivity"] = float(recall_score(yte, hard, zero_division=0))
        spec["test_specificity"] = float(tn / max((yte == 0).sum(), 1))
        spec["test_precision"] = float(precision_score(yte, hard, zero_division=0))
        spec["test_f1"] = float(f1_score(yte, hard, zero_division=0))
        spec["test_accuracy"] = float(accuracy_score(yte, hard))
        spec["accuracy_rule"] = "p >= t"
        if tag == "default":
            spec["ties_at_threshold"] = int((p1 == t).sum())
            spec["agrees_with_predict"] = bool(
                spec["test_accuracy"] == res["test_accuracy"])
    return res, pred, p1, secs


# The pre-declared comparison family. Six contrasts, fixed before any number
# was inspected, each a single change, Holm-corrected together. Anything else in
# the output is descriptive and is labelled as such.
#
# The fifth contrast is a SWAP, not a removal: L1 = {RBC, MCV, MCH} and
# L4 = {RBC, MCV, RDW}, so MCH goes out and RDW comes in and the feature count
# is held at three. That is the fair version of "what was the identity worth",
# because a bare removal would also shrink the model. It does mean the gap
# mixes losing MCH with gaining RDW; ablation.py's 2x2 (L1/L1d/L4/L6) is the
# uncontaminated decomposition and is where that separation is reported.
# The `meaning` strings name the comparison, not the outcome. They used to read
# "rf beats gradient boosting" and so on, which was a result written into the
# label of a test declared before any result existed - and on this cohort one of
# them is false: rf against xgboost is a null result on accuracy and a
# significant LOSS on PR-AUC, the metric the models were selected on, and it
# survives Holm. A pre-declared family cannot carry its own conclusions in its
# labels; the signed delta and the Holm p-value are where the direction is read.
#
# Deliberately no numbers in this note. It used to quote the delta and both
# p-values to five decimals, including one line correcting an earlier version of
# itself that had printed the RAW p under the corrected name - in the file whose
# own significance block asserts that no raw p is quoted anywhere as if it were
# corrected. Both the raw and the Holm value moved again when the replicate count
# was unified, and the comment did not. They live in
# significance.family[*].p_holm; that is the only copy.
FAMILY = [
    ("rf", HEADLINE, "majority", HEADLINE, "rf against the majority baseline"),
    ("rf", HEADLINE, "logreg", HEADLINE, "rf against logistic regression"),
    ("rf", HEADLINE, "dtree", HEADLINE, "rf against a single tree"),
    ("rf", HEADLINE, "xgboost", HEADLINE, "rf against gradient boosting"),
    ("rf", "L1_paper_leaky", "rf", "L4_identity_free",
     "MCH swapped for RDW, feature count fixed"),
    ("rf", "L1_paper_leaky", "rf", HEADLINE,
     "submitted model vs defensible model"),
]


def contrast(yte, pa, pb, qa, qb, strata, psu):
    """One paired contrast: design-based accuracy and PR-AUC, plus McNemar."""
    out = {"accuracy": design_test(
        lambda w: (accuracy_score(yte, pa, sample_weight=w)
                   - accuracy_score(yte, pb, sample_weight=w)), strata, psu)}
    rse = row_se(
        lambda i: (accuracy_score(yte, pa) - accuracy_score(yte, pb)) if i is None
        else (accuracy_score(yte[i], pa[i]) - accuracy_score(yte[i], pb[i])),
        len(yte))
    out["accuracy"]["row_se"] = rse
    out["accuracy"]["se_ratio_design_over_row"] = round(
        out["accuracy"]["se"] / max(rse, 1e-12), 2)
    out["mcnemar"] = mcnemar_test(yte, pa, pb)
    if qa is not None and qb is not None:
        out["pr_auc"] = design_test(
            lambda w: (average_precision_score(yte, qa, sample_weight=w)
                       - average_precision_score(yte, qb, sample_weight=w)),
            strata, psu)
    return out


def main():
    df, att = dp.load_cohort()
    tr, va, te = dp.split3(df)
    strata, psu, _ = dp.design_arrays(df, te)
    y = df["Anemia"].to_numpy()
    ytr, yva, yte = y[tr], y[va], y[te]
    nir = float(max(yte.mean(), 1 - yte.mean()))

    print(f"cohort {att['n_analysis']:,}   train {len(tr):,}   "
          f"val {len(va):,}   test {len(te):,}")
    print(f"anemia prevalence  train {ytr.mean()*100:.2f}%  "
          f"val {yva.mean()*100:.2f}%  test {yte.mean()*100:.2f}%")
    print(f"test no-information rate {nir*100:.2f}%   "
          f"PR-AUC no-skill {yte.mean():.4f}")
    print(f"test design: {len(set(zip(strata, psu)))} PSUs in "
          f"{len(set(strata))} strata, {dp.design_df(strata, psu)} df")
    print(f"selection rule declared up front: highest {SELECT_METRIC}; "
          f"{LIKE_FOR_LIKE} reported alongside as the like-for-like\n")

    out = {"cohort": att,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "no_information_rate": nir,
           "pr_auc_no_skill": float(yte.mean()),
           "seed": SEED, "n_folds": N_FOLDS, "n_cv_repeats": N_REPEATS,
           "n_replicates": N_REP,
           "selection_metric": SELECT_METRIC,
           "like_for_like_model": LIKE_FOR_LIKE,
           "variance_estimator": (
               f"Rao-Wu rescaled bootstrap, {N_REP} replicates, "
               f"{dp.design_df(strata, psu)} df"),
           "estimand": (
               "UNWEIGHTED within-sample performance, with stratum- and "
               "PSU-aware uncertainty. The Rao-Wu replicate weights are built "
               "on unit base weights (dataprep.rao_wu_weights(base=None)), so "
               "every interval and p-value here is for the accuracy of this "
               "classifier on rows like these, NOT for what it would score in "
               "the US population. That is the right estimand for a comparison "
               "between two models on the same rows: the survey weights would "
               "change the question, not sharpen the answer. The "
               "population-weighted estimates are in weighted.py / "
               "results/weighted.json, which passes MECWT as the base weight. "
               "The design effect of 4.00 reported there belongs to the "
               "weighted prevalence and does not transfer to these paired "
               "differences; ablation.py measures the actual Rao-Wu penalty on "
               "paired accuracy differences at 0.90x the i.i.d.-row SE."),
           "binary": {}, "severity": {}, "significance": {}}

    preds, probas = {}, {}          # (rung, model) -> arrays, for paired tests

    # ------------------------------------------------------ binary, 3 rungs --
    for rung in RUNGS:
        cols = dp.LADDER[rung]
        X = df[cols].to_numpy()
        out["binary"][rung] = {"features": list(cols), "models": {}}
        print(f"=== {rung}  {list(cols)}")
        for name, est in models().items():
            res, pred, p1, secs = evaluate(est, X, y, tr, va, te, strata, psu)
            out["binary"][rung]["models"][name] = res
            preds[(rung, name)] = pred
            probas[(rung, name)] = p1
            print(f"    {name:<11} cv {res['cv_accuracy_mean']:.4f}"
                  f"+/-{res['cv_accuracy_sd']:.4f}  val PR-AUC "
                  f"{res['val_pr_auc']:.4f}  test acc {res['test_accuracy']:.4f}"
                  f"  test PR-AUC {res['pr_auc']:.4f}"
                  f"  recall {res['recall']:.4f}"
                  f"  brier {res['calibration']['brier']:.4f}"
                  f"  [{secs}s]")
        print()

    # ---------------------------------------------- model selection, honest --
    # Chosen on validation, by the metric declared in SELECT_METRIC, over the
    # real estimators only. The two Dummy baselines are excluded because
    # "the best model was the majority-class baseline" is a different finding
    # and is reported separately if it happens.
    rows = out["binary"][HEADLINE]["models"]
    real = [m for m in rows if m not in BASELINES]
    winner = max(real, key=lambda m: rows[m][SELECT_METRIC])
    out["selected_model"] = winner
    out["selection_table"] = {m: rows[m][SELECT_METRIC] for m in real}
    print(f"SELECTION on the validation split at {HEADLINE}")
    for m in sorted(real, key=lambda k: -rows[k][SELECT_METRIC]):
        star = "  <- selected" if m == winner else ""
        lfl = "  (like-for-like with the manuscript)" if m == LIKE_FOR_LIKE else ""
        print(f"  {m:<12}{SELECT_METRIC} {rows[m][SELECT_METRIC]:.4f}{star}{lfl}")
    print("  test-set numbers below were not consulted in this choice")
    print(f"  {winner} wins on validation; {LIKE_FOR_LIKE} is carried through "
          f"every leakage contrast so the comparison with the")
    print("  submitted manuscript is like-for-like. Both are reported.\n")

    # ----------------------------------------------------- severity, 2 rungs --
    ysev = df["Severity"].to_numpy()
    for rung in ("L1_paper_leaky", HEADLINE):
        cols = dp.LADDER[rung]
        X = df[cols].to_numpy()
        out["severity"][rung] = {"features": list(cols), "models": {}}
        print(f"=== {rung} severity  {list(cols)}")
        for name, est in models().items():
            res, pred, _, secs = evaluate(est, X, ysev, tr, va, te, strata, psu,
                                    multiclass=True)
            out["severity"][rung]["models"][name] = res
            preds[("sev:" + rung, name)] = pred
            print(f"    {name:<11} acc {res['test_accuracy']:.4f}"
                  f"  macro-F1 {res['macro_f1']:.4f}"
                  f"  per-class recall "
                  f"{[round(v,3) for v in res['per_class_recall']]}"
                  f"  [{secs}s]")
        print()

    # --------------------------------------------------------- significance --
    print("PRE-DECLARED FAMILY OF SIX, Holm-corrected together")
    fam = []
    for ma, ra, mb, rb, what in FAMILY:
        fam.append(contrast(yte, preds[(ra, ma)], preds[(rb, mb)],
                            probas[(ra, ma)], probas[(rb, mb)], strata, psu))
    # Accuracy is the primary endpoint and PR-AUC the secondary one. Each is
    # Holm-corrected within its own family of six rather than pooling twelve
    # tests: pooling would penalise the primary endpoint for questions it was
    # not asked. Both corrections are reported, and no raw p is quoted anywhere
    # as if it were corrected.
    adj = holm([c["accuracy"]["p_value"] for c in fam])
    adj_pr = holm([c["pr_auc"]["p_value"] for c in fam])
    out["significance"]["correction"] = {
        "method": "Holm step-down, within endpoint",
        "primary_endpoint": "accuracy",
        "secondary_endpoint": "pr_auc",
        "n_tests_per_endpoint": len(FAMILY)}
    out["significance"]["family"] = []
    print(f"  {'contrast':<44}{'acc diff':>10}{'design 95% CI':>20}"
          f"{'p':>8}{'Holm':>8} | {'McNemar p':>11}{'SE x row':>9}")
    for (ma, ra, mb, rb, what), c, pa, ppr in zip(FAMILY, fam, adj, adj_pr):
        a = c["accuracy"]
        c.update({"a": f"{ma}@{ra}", "b": f"{mb}@{rb}", "meaning": what,
                  "p_holm": pa, "significant_holm_05": bool(pa < 0.05)})
        c["pr_auc"]["p_holm"] = ppr
        c["pr_auc"]["significant_holm_05"] = bool(ppr < 0.05)
        out["significance"]["family"].append(c)
        ci = f"[{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}]"
        mark = "*" if pa < 0.05 else " "
        print(f"  {what:<44}{a['value']:+10.4f}{ci:>20}{a['p_value']:>8.4f}"
              f"{pa:>8.4f}{mark}|{c['mcnemar']['p_value']:>11.2e}"
              f"{a['se_ratio_design_over_row']:>9.2f}")
    print("  * = survives Holm at 0.05.  'p' is a design-based t test on "
          f"{dp.design_df(strata, psu)} df;")
    print("  'McNemar p' is the same contrast under the independent-rows "
          "assumption the")
    print("  literature makes. Both are printed because the direction of the "
          "difference is")
    print("  an empirical question, not something a methods sentence can settle.")
    ratios = [c["accuracy"]["se_ratio_design_over_row"]
              for c in out["significance"]["family"]]
    n_mc_smaller = sum(c["mcnemar"]["p_value"] < c["accuracy"]["p_value"]
                       for c in out["significance"]["family"])
    n_agree = sum((c["mcnemar"]["p_value"] < 0.05)
                  == (c["accuracy"]["p_value"] < 0.05)
                  for c in out["significance"]["family"])
    out["significance"]["median_se_ratio_design_over_row"] = float(
        np.median(ratios))
    out["significance"]["mcnemar_vs_design"] = {
        "n_mcnemar_smaller": int(n_mc_smaller),
        "n_same_conclusion_at_05": int(n_agree),
        "of": len(FAMILY),
        "compared": "uncorrected McNemar p vs uncorrected design p, so the "
                    "only thing varying is the variance estimator; comparing "
                    "either one against the Holm-corrected flag would mix in "
                    "the multiplicity correction and measure two things at "
                    "once",
        "note": "For an unweighted PAIRED accuracy difference the clustering "
                "penalty is small: design standard errors are close to the "
                "i.i.d. ones. The design effect near 4 reported in "
                "weighted.py is for a weighted PREVALENCE, where unequal "
                "weights dominate, and it does not transfer to this contrast. "
                "Stating that honestly is better than implying every "
                "unweighted comparison in this literature is invalidated."}
    print(f"  design standard errors are {np.median(ratios):.2f}x the i.i.d. "
          f"row bootstrap, median over the six;")
    print(f"  McNemar is the smaller p in {n_mc_smaller} of {len(FAMILY)}, and "
          f"the two uncorrected tests agree")
    print(f"  on {n_agree} of {len(FAMILY)} conclusions at 0.05 (both "
          f"uncorrected, so this isolates the variance")
    print("  estimator). So clustering does not overturn these paired "
          "contrasts. The design")
    print("  effect near 4 in weighted.py is for a weighted PREVALENCE, where "
          "unequal weights")
    print("  dominate; it does not transfer to an unweighted within-sample "
          "accuracy difference.")

    # descriptive, outside the family: does the selected model beat RF?
    if winner != LIKE_FOR_LIKE:
        c = contrast(yte, preds[(HEADLINE, winner)],
                     preds[(HEADLINE, LIKE_FOR_LIKE)],
                     probas[(HEADLINE, winner)],
                     probas[(HEADLINE, LIKE_FOR_LIKE)], strata, psu)
        c["meaning"] = (f"{winner} minus {LIKE_FOR_LIKE} at {HEADLINE} "
                        "(descriptive, outside the pre-declared family)")
        out["significance"]["selected_vs_like_for_like"] = c
        a, q = c["accuracy"], c["pr_auc"]
        print(f"\n  descriptive, not in the family: {winner} minus "
              f"{LIKE_FOR_LIKE} at {HEADLINE}")
        print(f"    accuracy {a['value']:+.4f} [{a['ci95'][0]:+.4f},"
              f"{a['ci95'][1]:+.4f}] p {a['p_value']:.4f}   "
              f"PR-AUC {q['value']:+.4f} [{q['ci95'][0]:+.4f},"
              f"{q['ci95'][1]:+.4f}] p {q['p_value']:.4f}")
    else:
        out["significance"]["selected_vs_like_for_like"] = None
        print(f"\n  the validation winner IS {LIKE_FOR_LIKE}, so the "
              f"like-for-like model and the selected model coincide")

    # ---------------------------------------------------------- paper table --
    print(f"\n\nTABLE: comparators at {HEADLINE} (the defensible model)")
    hdr = (f"{'model':<12}{'CV acc (20 fits)':>18}{'val PR-AUC':>11}"
           f"{'test acc':>9}{'95% CI':>17}{'bal acc':>8}{'ROC':>7}"
           f"{'PR-AUC':>8}{'recall':>8}{'F1':>7}{'Brier':>8}")
    print(hdr)
    print("-" * len(hdr))
    order = sorted(rows, key=lambda k: -rows[k]["val_pr_auc"])
    for name in order:
        r = rows[name]
        ci = f"[{r['test_accuracy_ci95'][0]:.3f},{r['test_accuracy_ci95'][1]:.3f}]"
        tag = "*" if name == winner else ("+" if name == LIKE_FOR_LIKE else " ")
        print(f"{name:<11}{tag}{r['cv_accuracy_mean']:.4f}"
              f"+/-{r['cv_accuracy_sd']:.4f}{r['val_pr_auc']:11.4f}"
              f"{r['test_accuracy']:9.4f}{ci:>17}"
              f"{r['test_balanced_accuracy']:8.4f}{r['roc_auc']:7.3f}"
              f"{r['pr_auc']:8.4f}{r['recall']:8.4f}{r['f1']:7.4f}"
              f"{r['calibration']['brier']:8.4f}")
    print("-" * len(hdr))
    print(f"{'no-information':<12}{'-':>18}{yte.mean():11.4f}{nir:9.4f}"
          f"{'-':>17}{0.5:8.4f}{0.5:7.3f}{yte.mean():8.4f}{0.0:8.4f}"
          f"{0.0:7.4f}{'-':>8}")
    print("* = selected on validation   + = like-for-like with the manuscript")
    print("CIs are design-based (Rao-Wu replicate weights, "
          f"{dp.design_df(strata, psu)} df). PR-AUC's no-skill value is the")
    print(f"prevalence, {yte.mean():.4f}, not the no-information rate "
          f"{nir:.4f}.")
    best_acc = max(rows[m]["test_accuracy"] for m in real)
    best_pr = max(rows[m]["pr_auc"] for m in real)
    print(f"Accuracy has almost no dynamic range here: the majority baseline "
          f"scores {nir:.4f} against")
    print(f"{best_acc:.4f} for the best model, a span of "
          f"{(best_acc-nir)*100:.2f} points, while PR-AUC spans "
          f"{(best_pr-yte.mean())*100:.1f}")
    print("points from the same two rows. A paper that reports only accuracy is "
          "reporting the")
    print("metric least able to distinguish a model from a constant.")

    # ----------------------------------------------------- operating points --
    print(f"\n\nTABLE: operating points for {LIKE_FOR_LIKE} at {HEADLINE}, "
          f"chosen on validation")
    print(f"  {'rule':<44}{'thr':>7}{'sens':>8}{'spec':>8}{'prec':>7}{'F1':>7}"
          f"{'acc':>8}")
    for tag, s in rows[LIKE_FOR_LIKE]["thresholds"].items():
        # A None threshold is a result, not a gap: pick_thresholds returns one
        # when roc_curve offers nothing but its infinite endpoint, or when no
        # validation threshold reaches the required specificity. evaluate() then
        # scores nothing for that row, so this used to format None with :7.3f and
        # read test_sensitivity out of a dict that has no such key - a crash in
        # the printing of a table, at the end of the longest stage in the
        # pipeline. It cannot fire on this cohort, where all four rules resolve;
        # it fires on a rung or a subgroup where a model's scores are constant.
        if s["threshold"] is None:
            print(f"  {s['rule']:<44}{'-':>7}   {s.get('unavailable', 'no usable operating point')}")
            continue
        print(f"  {s['rule']:<44}{s['threshold']:7.3f}"
              f"{s['test_sensitivity']:8.4f}{s['test_specificity']:8.4f}"
              f"{s['test_precision']:7.4f}{s['test_f1']:7.4f}"
              f"{s['test_accuracy']:8.4f}")
    print("  thresholds were tuned on validation and applied unchanged to test;")
    print("  the 0.5 row is the default the manuscript reported without saying so")
    # The default row's accuracy is computed as p >= 0.5, and est.predict() breaks
    # a tie at exactly 0.5 the other way, toward class 0. When the two disagree,
    # the same model's accuracy appears in two tables in this run under two
    # numbers. evaluate() has recorded the comparison as `agrees_with_predict`
    # since it was written and nothing ever printed it, while the comment at the
    # recording site said the gap was "visible". For rf@L6 it is 0.9667 in the
    # headline table and 0.9659 here. Printed now, both ways, so a reader who
    # notices the discrepancy finds it named rather than having to reconcile it.
    dflt = rows[LIKE_FOR_LIKE]["thresholds"].get("default", {})
    if "agrees_with_predict" in dflt:
        if dflt["agrees_with_predict"]:
            print(f"  the 0.5 row agrees with est.predict() exactly "
                  f"({dflt['ties_at_threshold']} rows sit at p = 0.5)")
        else:
            print(f"  NOTE: the 0.5 row is {dflt['test_accuracy']:.4f} where the "
                  f"headline table says "
                  f"{rows[LIKE_FOR_LIKE]['test_accuracy']:.4f}. Same model, two "
                  f"rules:")
            print("  this row is p >= 0.5, the headline is est.predict(), which "
                  "sends a tie at")
            print(f"  exactly 0.5 to class 0. {dflt['ties_at_threshold']} test "
                  f"rows sit on the tie.")

    # -------------------------------------------- rung vs algorithm, 9 rows --
    print("\n\nTABLE: what the rung does vs what the algorithm does")
    hdr2 = (f"{'model':<12}{'L1':>10}{'L4':>10}{'L6':>10}"
            f"{'L1-L4':>9}{'L1-L6':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    per_rung = {r: [] for r in RUNGS}
    for name in models():
        a = {r: out["binary"][r]["models"][name]["test_accuracy"] for r in RUNGS}
        if name not in BASELINES:
            for r in RUNGS:
                per_rung[r].append(a[r])
        print(f"{name:<12}{a[RUNGS[0]]:10.4f}{a[RUNGS[1]]:10.4f}"
              f"{a[RUNGS[2]]:10.4f}"
              f"{a[RUNGS[0]] - a[RUNGS[1]]:+9.4f}"
              f"{a[RUNGS[0]] - a[RUNGS[2]]:+9.4f}")
    print("-" * len(hdr2))
    spread = {r: max(v) - min(v) for r, v in per_rung.items()}
    out["accuracy_spread_across_models"] = {r: float(s) for r, s in spread.items()}
    print("spread across the 7 real models  "
          + "  ".join(f"{r.split('_')[0]} {spread[r]:.4f}" for r in RUNGS))

    # Is the identity gap a property of the feature set or of one algorithm?
    # The answer is the SIGN COUNT, not the spread: the spread across estimators
    # is larger than the gap, so a claim that the feature set dominates the
    # algorithm would be false. What is true, and stronger, is that the gap has
    # the same sign for every estimator.
    gaps_acc = {m: (out["binary"]["L1_paper_leaky"]["models"][m]["test_accuracy"]
                    - out["binary"]["L4_identity_free"]["models"][m]["test_accuracy"])
                for m in real}
    gaps_pr = {m: (out["binary"]["L1_paper_leaky"]["models"][m]["pr_auc"]
                   - out["binary"]["L4_identity_free"]["models"][m]["pr_auc"])
               for m in real}
    same_acc = sum(v > 0 for v in gaps_acc.values())
    same_pr = sum(v > 0 for v in gaps_pr.values())
    out["identity_gap_across_estimators"] = {
        "accuracy": {m: float(v) for m, v in gaps_acc.items()},
        "pr_auc": {m: float(v) for m, v in gaps_pr.items()},
        "n_positive_accuracy": int(same_acc),
        "n_positive_pr_auc": int(same_pr),
        "n_estimators": len(real),
        "mean_gap_accuracy": float(np.mean(list(gaps_acc.values()))),
        "mean_gap_pr_auc": float(np.mean(list(gaps_pr.values()))),
    }

    # look the contrast up by what it is, not by where it sits in the list:
    # inserting a contrast must not silently re-point this paragraph
    _want = (LIKE_FOR_LIKE, "L1_paper_leaky", LIKE_FOR_LIKE, "L4_identity_free")
    _at = [i for i, f in enumerate(FAMILY) if f[:4] == _want]
    assert len(_at) == 1, f"expected exactly one {_want} contrast, found {_at}"
    lk = out["significance"]["family"][_at[0]]
    print(f"\nSwapping the MCH identity for RDW costs {LIKE_FOR_LIKE} "
          f"{lk['accuracy']['value']*100:+.2f} points of accuracy "
          f"(design p {lk['accuracy']['p_value']:.3f}, Holm "
          f"{lk['p_holm']:.3f}) but")
    print(f"{lk['pr_auc']['value']*100:+.2f} points of PR-AUC "
          f"(p {lk['pr_auc']['p_value']:.3f}, Holm {lk['pr_auc']['p_holm']:.3f}). "
          f"Accuracy is the least sensitive metric")
    print("to the leak, which is why an accuracy-only paper cannot see it. "
          "Feature count is")
    print("held at three across the swap; ablation.py's 2x2 separates losing "
          "MCH from")
    print("gaining RDW.")
    print(f"\nThe gap is positive for {same_acc} of {len(real)} estimators on "
          f"accuracy (mean "
          f"{np.mean(list(gaps_acc.values()))*100:+.2f} points) and "
          f"{same_pr} of {len(real)}")
    print(f"on PR-AUC (mean {np.mean(list(gaps_pr.values()))*100:+.2f} points), "
          f"so it is a property of the feature")
    print("set rather than of one algorithm. Note what is NOT claimed: the "
          "spread across")
    print(f"estimators ({spread[RUNGS[0]]:.4f} at L1, {spread[RUNGS[2]]:.4f} at "
          f"L6) is larger than the rung gap, so the")
    print("algorithm matters too. The point is that no choice of algorithm "
          "removes the leak.")

    out["answers"] = {
        # The count is len(real), not the literal "Nine" this string used to
        # carry, and not len(rows) either. Nine is the number of entries in
        # `models` - seven estimators plus the two Dummy baselines - and the
        # baselines were never selection candidates: `winner` is a max over
        # `real`, and `selection_table` is built from `real`, so the search was
        # over seven. The sentence claimed a selection over nine models and then
        # quoted a search over seven, in the answer to the reviewer question that
        # asks how the model was chosen. `rows` is the whole models dict and would
        # have reproduced the same nine. The neighbouring spread_vs_rung key had
        # been deriving this count from len(real) all along, which is how the
        # disagreement became visible.
        "why_random_forest": (
            f"It is not asserted. {len(real)} estimators were run on identical "
            f"folds and rows (plus {len(BASELINES)} trivial baselines, which were "
            f"not selection candidates); {winner} had the highest validation "
            f"{SELECT_METRIC} ({rows[winner][SELECT_METRIC]:.4f}) and "
            f"{LIKE_FOR_LIKE} had "
            f"{rows[LIKE_FOR_LIKE][SELECT_METRIC]:.4f}. Random Forest is "
            f"carried through the leakage contrasts because the manuscript "
            f"under revision used one, so the comparison is like-for-like; the "
            f"validation winner is reported next to it."),
        "spread_vs_rung": (
            f"The identity gap L1 minus L4 is positive for {same_acc} of "
            f"{len(real)} estimators on accuracy (mean "
            f"{np.mean(list(gaps_acc.values()))*100:+.2f} points) and "
            f"{same_pr} of {len(real)} on PR-AUC (mean "
            f"{np.mean(list(gaps_pr.values()))*100:+.2f} points), so it is a "
            f"property of the feature set and no algorithm removes it. It is "
            f"NOT claimed that the feature set matters more than the "
            f"algorithm: accuracy spans {spread[HEADLINE]:.4f} across the "
            f"seven real estimators at {HEADLINE}, which is larger than the "
            f"rung gap."),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
