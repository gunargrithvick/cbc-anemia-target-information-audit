"""
Step 3 - the leakage ablation ladder.

For every rung of dataprep.LADDER this script answers two questions at once:

  1. How well can hemoglobin be RECONSTRUCTED from this feature set?
     (the mechanism: if Hb is recoverable, the label is still in the inputs)
  2. How well does a Random Forest CLASSIFY anemia from this feature set?
     (the consequence)

Putting them side by side is the paper's central result. The result has to be
stated carefully, though, and the first version of this script did not: accuracy
tracks recoverability WITHIN A FIXED FEATURE SCOPE. It is not a law that holds
across the whole ladder, because L0 hands the model hemoglobin verbatim and
still scores below L6, which has demographics instead. So the claim is measured
here rather than asserted: rank correlation inside the no-demographics rungs,
rank correlation over everything, and both printed.

Seven things are deliberate.

  Every model here is fitted on the 60% training split only, so benchmark.py's
  validation split stays genuinely unused and every number in the paper comes
  from the same fit. Because that rule changes one contrast's p-value, the
  contrast is also reported under an 80% fit; see train_size_sensitivity.

  Every rung shares ONE cohort, complete on the union of all ladder features,
  because rungs fitted on different rows are not comparable and comparing rungs
  is the study. That choice looked like it cost rows a narrow rung never needed
  to lose; complete_case_sensitivity measures the bill and it is zero, for a
  reason worth knowing - see that function.

  Variance is design-based, from Rao-Wu rescaled bootstrap replicate weights on
  25 degrees of freedom (PSUs minus strata), not from resampling rows. Three
  scales are printed side by side for every contrast -- i.i.d. rows, the naive
  n_h-of-n_h cluster bootstrap, and Rao-Wu -- because the naive cluster scheme
  is itself wrong at two PSUs per stratum, where its variance expectation is
  (n_h - 1)/n_h of the truth. Asserting that a survey design widens intervals
  is cheap; measuring by how much, and in which direction, is the point.

  The comparisons are a pre-declared family of six, Holm-corrected. Running
  every pair and reporting whichever came out under 0.05 is the same practice
  this paper criticises.

  Recoverability is the BEST of a small family of estimators, not the score of
  one Random Forest. A log-linear least-squares fit beats the forest on several
  rungs, and reporting only the forest would understate how much of the label
  survives. "Best" means best on the VALIDATION block: the family is fitted on
  train, ranked on validation, and only the winner is scored on test. Selecting
  the winner on test would have inflated recoverability, which is the direction
  that suits this paper's argument, so the JSON reports both that oracle
  (best_on_test) and what it would have added (selection_optimism_gdl).

  Label agreement is computed on the test rows for every route, exact or
  learned, so the column has one denominator.

  Every rung is also evaluated on three NHANES cycles the model never saw,
  which is the only evidence that the ladder is a property of the CBC rather
  than of one sample.

Outputs
  results/ablation.json
  stdout    the tables to put in the paper

Run:  python src/ablation.py
"""

import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats

import dataprep as dp
from download_data import CYCLES, PRIMARY
from paths import RESULTS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = dp.SEED
N_FOLDS = dp.N_FOLDS       # one place, dataprep; see the note there
N_REP = dp.N_REP           # one place, dataprep; see the note there
OUT_PATH = RESULTS / "ablation.json"

# The pre-declared comparison family. Six contrasts, fixed before the numbers
# were looked at, each one a single change. Holm-corrected together.
CONTRASTS = [
    ("L0_hgb_direct", "L1_paper_leaky", "hemoglobin verbatim -> identity route"),
    ("L1_paper_leaky", "L4_identity_free", "remove the identity, no demographics"),
    ("L1_paper_leaky_demo", "L6_identity_free_demo", "remove the identity, with demographics"),
    ("L1_paper_leaky", "L1_paper_leaky_demo", "add demographics, identity present"),
    ("L4_identity_free", "L6_identity_free_demo", "add demographics, identity gone"),
    ("L6_identity_free_demo", "L7_hgb_demo", "defensible model -> the complete leak"),
]


def forest(n_estimators=dp.N_TREES):
    """The manuscript's classifier, minus the pointless StandardScaler."""
    return RandomForestClassifier(n_estimators=n_estimators,
                                  class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def holm(pvals):
    """
    Delegates to dp.holm. See its docstring for why there is only one copy now.

    Kept as a one-line wrapper rather than deleted so that `holm(...)` at the two
    call sites below still reads the same, and so that anything that imported
    ablation.holm keeps working.
    """
    return dp.holm(pvals)


_REP_CACHE = {}


def replicate_weights(strata, psu, n=N_REP, seed=SEED):
    """
    n Rao-Wu replicate weight vectors for one design, built once and reused.

    Every statistic in this file is evaluated on the same test rows, so the
    replicate weights are a property of the design and not of the model. Sharing
    them across rungs is also what makes the contrasts comparable: two rungs
    are reweighted identically in replicate b.
    """
    key = (strata.tobytes(), psu.tobytes(), n, seed)
    if key not in _REP_CACHE:
        rng = np.random.default_rng(seed)
        _REP_CACHE[key] = np.stack(
            [dp.rao_wu_weights(strata, psu, rng, base=None) for _ in range(n)])
    return _REP_CACHE[key]


def cluster_ci(y_true, y_pred, strata, psu, stat=accuracy_score,
               n=N_REP, alpha=0.05, seed=SEED):
    """
    Design-based CI from Rao-Wu rescaled bootstrap replicate weights.

    theta-hat is computed once on the sample. The replicate weights give the
    variance, the degrees of freedom are (PSUs - strata) = 25 rather than
    n - 1 = 2,431, and the interval is a t interval on that df. Both of those
    corrections matter and neither appears in any of the 20 papers reviewed.

    Returns (lo, hi, se, degenerate). The fourth value is the one that was
    missing. L0_hgb_direct and L7_hgb_demo both classify every test row
    correctly - hemoglobin against a threshold derived from hemoglobin - so every
    replicate returns exactly 1.0, se is 0, and the interval collapses to
    [1.0, 1.0]. That was published as a confidence interval, and a reader is
    entitled to read [1.000, 1.000] as "certain to five decimal places" when what
    happened is that a degenerate statistic has no sampling distribution to
    estimate. Same for recall on those rungs, and for the majority baseline's
    recall, which is [0.0, 0.0] for the mirror-image reason.

    The interval is still returned, because the arithmetic is what it is and
    truncating it to None would break every caller that formats it. The fourth
    value says not to trust it, and callers write it into the JSON beside the
    interval.
    """
    theta = float(stat(y_true, y_pred))
    df = dp.design_df(strata, psu)
    W = replicate_weights(strata, psu, n, seed)
    vals = np.array([stat(y_true, y_pred, sample_weight=w) for w in W])
    se = float(np.sqrt(np.mean((vals - theta) ** 2)))
    t = stats.t.ppf(1 - alpha / 2, df)
    degenerate = None
    if se == 0.0:
        degenerate = (f"every one of the {len(W)} replicates returned "
                      f"{theta:.6g}, so the replicate spread is exactly zero and "
                      f"this interval is a point, not a measurement of "
                      f"uncertainty. It happens when the statistic is at a "
                      f"boundary the design cannot move it off - a rung that is "
                      f"right on every row, or a baseline that is wrong on every "
                      f"positive.")
    return theta - t * se, theta + t * se, se, degenerate


def _acc_diff(y, p_before, p_after, w=None):
    """
    accuracy(after) - accuracy(before). Callers pass (before, after).

    The sign convention matters and it used to be inverted. A CONTRASTS entry
    names a transition - "add demographics, identity gone" for L4 -> L6 - and
    the function returned acc(L4) - acc(L6), so adding demographics reported as
    -0.0317: a gain printed as a loss, and fig4 drew the arrow that way. Every
    difference in this file now carries the sign of the transition its own label
    describes. Two-sided p-values are unaffected; deltas and CIs all flip.
    """
    return (accuracy_score(y, p_after, sample_weight=w)
            - accuracy_score(y, p_before, sample_weight=w))


def paired_cluster_test(y, pred_before, pred_after, strata, psu,
                        n=N_REP, seed=SEED):
    """
    Is rung A's accuracy different from rung B's, on the same test rows?

    Argument order is (before, after) and the returned delta_accuracy is
    after - before, so it reads in the direction the contrast's label names.

    Design-based paired test. Each Rao-Wu replicate reweights whole PSUs and
    recomputes BOTH accuracies under the same weights, so the correlation
    between the two models is preserved and only the design contributes to the
    spread. The statistic is delta / se on (PSUs - strata) degrees of freedom.

    A t test on 25 df rather than a percentile interval is deliberate: with 24
    strata the replicate distribution is coarse, and reading its 2.5th
    percentile would be over-precise about its own tails.
    """
    obs = _acc_diff(y, pred_before, pred_after)
    df = dp.design_df(strata, psu)
    W = replicate_weights(strata, psu, n, seed)
    vals = np.array([_acc_diff(y, pred_before, pred_after, w) for w in W])
    se = float(np.sqrt(np.mean((vals - obs) ** 2)))
    # 0/0 is not infinite evidence, and neither is a nonzero delta that no
    # replicate moved. dataprep.t_and_p owns both cases, so this file, benchmark.py
    # and weighted.py cannot answer them differently - which they used to.
    tp = dp.t_and_p(obs, se, df)
    tcrit = stats.t.ppf(0.975, df)
    return {"delta_accuracy": float(obs),
            "se": se,
            "df": df,
            "t": tp["t"],
            "degenerate": tp["degenerate"],
            "ci95": [float(obs - tcrit * se), float(obs + tcrit * se)],
            "p_value": tp["p_value"]}


def paired_naive_cluster_test(y, pred_a, pred_b, strata, psu, n=N_REP, seed=SEED):
    """The same contrast under the naive n_h-of-n_h cluster bootstrap."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = dp.psu_bootstrap(strata, psu, rng)
        diffs[i] = _acc_diff(y[idx], pred_a[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    side = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"se": float(diffs.std(ddof=1)), "ci95": [float(lo), float(hi)],
            "p_value": float(min(1.0, max(2.0 * side, 1.0 / n)))}


def paired_row_test(y, pred_a, pred_b, n=N_REP, seed=SEED):
    """
    The same contrast done the way the literature does it: rows i.i.d.

    Kept in the code and printed next to the design-based version, because "we
    used a survey-aware variance estimator" is a methods sentence nobody checks,
    whereas three standard errors for one contrast is an argument.
    """
    rng = np.random.default_rng(seed)
    k = len(y)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        diffs[i] = _acc_diff(y[idx], pred_a[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    side = min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"se": float(diffs.std(ddof=1)), "ci95": [float(lo), float(hi)],
            "p_value": float(min(1.0, max(2.0 * side, 1.0 / n)))}


def hb_recoverability(df, cols, tr, va, te):
    """
    How much of the label survives in this feature set?

    Estimators are fitted on train, COMPARED on validation, and the one that
    wins on validation is reported on test. Test rows are never used to choose
    anything. What is reported:

      exact_routes  EVERY closed-form analyser identity this rung leaves open,
                  ranked by validation MAE. A rung can leave more than one --
                  L2 leaves two, L0 leaves two - and an if/elif chain reported
                  only whichever the maintainer had written first.
      exact       the best of those, or null on a rung that leaves none. A
                  closed form has no fitted parameter, so its test score is not
                  optimistic in any case; it is scored on validation too only
                  so that all candidates are ranked on one scale.
      family      every learned estimator, each with its validation MAE and its
                  full test metrics: a Random Forest, ordinary least squares,
                  and least squares in logs. The log fit is not decoration --
                  every analyser identity is a product, so in logs the exact
                  routes are linear and OLS finds them. On the identity-free
                  rungs it beats the forest, and quoting only the forest would
                  understate residual leakage. Nothing is hidden: the whole
                  family is in the JSON whether it won or not.
      learned     the family member with the lowest VALIDATION MAE, reported on
                  test.
      selected    the same rule over exact and family together. This is the
                  number the ladder is plotted against, because the question is
                  what an adversary COULD recover, not what one particular
                  model did.
      best_on_test    the lowest test MAE over the same candidates, which is an
                  oracle: it is what selecting on test would have returned.
                  Kept, clearly labelled, so the size of the selection effect
                  is visible rather than argued about.
      selection_optimism_gdl    selected minus best_on_test, in g/dL, >= 0 by
                  construction. It is the amount the old test-selected number
                  overstated recoverability by.

    The direction matters here. Choosing the winner on test inflates apparent
    recoverability, which is the direction that flatters this paper's own
    leakage thesis, so the honest rule is the conservative one.

    Label agreement is on the test rows in every case, so the column has one
    denominator. It applies each person's own WHO cut-off to both the true and
    the reconstructed hemoglobin: that uses threshold information the rung does
    not have, and it is reported as what it is -- how often a reconstruction
    would reproduce the label, not something the rung's classifier achieves.
    """
    have = set(cols)
    hb = df[dp.HGB].to_numpy()
    thr = df["WHO_NORMAL"].to_numpy()
    truth = df["Anemia"].to_numpy()

    def score(est_te, tag):
        """
        Rounded to 6 decimals, which is four orders of magnitude finer than
        anything reported and stops the file re-hashing for nothing.

        The forest's MAE moved in its last two or three digits between two
        otherwise identical runs - reduction order inside a threaded ensemble,
        about 1e-16 relative. No reported statistic ever changed, but 25 fields
        differed, so run_manifest.json flagged an inert run as a changed one.
        """
        d = np.abs(est_te - hb[te])
        return {"estimator": tag,
                "mae_gdl": round(float(d.mean()), 6),
                "rmse_gdl": round(float(np.sqrt((d ** 2).mean())), 6),
                "pearson_r": round(float(np.corrcoef(hb[te], est_te)[0, 1]), 6),
                "label_agreement": round(float(
                    ((est_te < thr[te]).astype(int) == truth[te]).mean()), 6)}

    def val_mae(est_va):
        """The selection criterion, and the only thing validation is used for."""
        return round(float(np.abs(est_va - hb[va]).mean()), 6)

    out = {}

    # ---- closed form, if this rung leaves an identity intact ----------------
    #
    # Every covered route is computed, not the first one an if/elif chain
    # happens to reach. That chain was the earlier version and two of its five
    # branches were dead code: L2 is the only rung carrying MCHC and it carries
    # HCT too, so `MCHC*HCT/100` always fired before `MCHC*MCV*RBC/1000`; and
    # every rung carrying MCH also carries RBC, so `MCH*RBC/10` always fired
    # before `MCH*HCT/MCV`. Two identities were therefore asserted in the source,
    # tested nowhere, and reported never. Ordering them by validation MAE also
    # answers the question the section is actually asking - what an adversary
    # could recover - rather than which line a maintainer wrote first.
    routes = {
        "HGB present verbatim": ({dp.HGB}, lambda: hb),
        "MCH*RBC/10": ({dp.MCH, dp.RBC},
                       lambda: (df[dp.MCH] * df[dp.RBC] / 10.0).to_numpy()),
        "MCHC*HCT/100": ({dp.MCHC, dp.HCT},
                         lambda: (df[dp.MCHC] * df[dp.HCT] / 100.0).to_numpy()),
        "MCHC*MCV*RBC/1000": (
            {dp.MCHC, dp.MCV, dp.RBC},
            lambda: (df[dp.MCHC] * df[dp.MCV] * df[dp.RBC] / 1000.0).to_numpy()),
        # The fourth route, and the one that is easiest to miss: it needs
        # neither RBC nor MCHC. MAE 0.0315 g/dL, WHO-label agreement 0.9944.
        "MCH*HCT/MCV": ({dp.MCH, dp.HCT, dp.MCV},
                        lambda: (df[dp.MCH] * df[dp.HCT] / df[dp.MCV]).to_numpy()),
    }
    exact_all = []
    for tag, (need, fn) in routes.items():
        if not need <= have:
            continue
        est = fn()
        s = score(est[te], tag)
        s["route"] = tag
        s["mae_gdl_val"] = val_mae(est[va])
        exact_all.append(s)

    # This dict and identity_audit.exact_routes state the same identities twice
    # on purpose: they are definitional, and two independent spellings
    # cross-check each other. dp.HGB_ROUTES is the authoritative list, so a
    # route added there but not here fails loudly rather than silently.
    _covered = sum(1 for r in dp.HGB_ROUTES if r <= have)
    assert bool(exact_all) == (dp.HGB in have or _covered > 0), \
        f"dp.HGB_ROUTES and hb_recoverability disagree on {sorted(have)}"
    assert len(exact_all) == _covered + (1 if dp.HGB in have else 0), \
        f"route coverage mismatch on {sorted(have)}: {len(exact_all)} vs {_covered}"

    # exact_routes is every route this rung leaves open, ranked on validation.
    # exact is the winner, and keeps the shape the rest of the file expects.
    out["exact_routes"] = sorted(exact_all, key=lambda s_: s_["mae_gdl_val"])
    out["exact"] = out["exact_routes"][0] if exact_all else None

    # ---- learned family ----------------------------------------------------
    X = df[cols].to_numpy(dtype=float)
    fam = []

    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X[tr], hb[tr])
    s = score(rf.predict(X[te]), "random forest")
    s["mae_gdl_val"] = val_mae(rf.predict(X[va]))
    s["r2"] = round(float(rf.score(X[te], hb[te])), 6)
    fam.append(s)

    ols = LinearRegression().fit(X[tr], hb[tr])
    s = score(ols.predict(X[te]), "ols")
    s["mae_gdl_val"] = val_mae(ols.predict(X[va]))
    s["r2"] = round(float(ols.score(X[te], hb[te])), 6)
    fam.append(s)

    # Logs, because every analyser identity is a product and so is linear here.
    #
    # This used to be gated on (X > 0).all(), which dropped the log model from
    # every rung that carries demographics: PREGNANT is 0 for 12,080 of the
    # 12,156 rows, so one all-zero-containing column removed a whole estimator
    # with no trace in the JSON. That truncation inverted the ladder. L4 kept
    # its log fit at MAE 0.2494 while L6, which CONTAINS L4, was left with the
    # forest at 0.2640 - strictly more information scoring strictly worse, which
    # is impossible for a "best recoverable" quantity and was purely an artefact
    # of the gate. Only the strictly-positive columns are logged now, which is
    # the right treatment regardless: a 0/1 indicator is not a multiplicative
    # factor in any identity. Which columns were logged is recorded.
    #
    # dp.NOMINAL_FEATURES is excluded on top of the positivity test, because
    # RIAGENDR is strictly positive and is still a code: 1 and 2 name two groups
    # and the 2 is not twice the 1. Logging it put a coefficient on log(sex) into
    # the JSON tagged `is_exponent: true`, which asserts that hemoglobin is
    # proportional to sex^-0.0121. Excluding it moves no fitted value at all -
    # the log of a two-valued column is an affine function of it and spans the
    # same space - so every published recoverability figure is unchanged and only
    # the claim attached to it is.
    pos = (X > 0).all(axis=0) & np.array(
        [c not in dp.NOMINAL_FEATURES for c in cols])

    def logged(rows):
        d = X[rows].copy()
        d[:, pos] = np.log(d[:, pos])
        return d

    lg = LinearRegression().fit(logged(tr), np.log(hb[tr]))
    est = np.exp(lg.predict(logged(te)))
    s = score(est, "ols in logs")
    s["mae_gdl_val"] = val_mae(np.exp(lg.predict(logged(va))))
    s["logged_columns"] = [c for c, k in zip(cols, pos) if k]
    s["untransformed_columns"] = [c for c, k in zip(cols, pos) if not k]
    s["coefficients"] = [
        {"column": c, "value": round(float(v), 4),
         "is_exponent": bool(k)}
        for c, v, k in zip(cols, lg.coef_, pos)]
    s["coefficient_note"] = (
        "A coefficient on a logged column is an exponent in the recovered "
        "product; a coefficient on an untransformed column is not.")
    fam.append(s)

    # Monotonicity is a property of "best recoverable", not a hope: a rung that
    # contains another cannot recover less. Violations here have always meant a
    # bug in the family, never a finding, so the invariant is checked in
    # tracking() where every rung's result is available.

    out["family"] = fam

    # ---- selection, on validation only -------------------------------------
    #
    # This used to be min(..., key="mae_gdl"), i.e. the argmin over TEST MAE,
    # which is selection on the test block. The bias it introduced ran toward
    # this paper's own conclusion - the winner-on-test always looks like more
    # recoverable hemoglobin than an honestly chosen estimator does - so it was
    # the one place in the study where a shortcut flattered the thesis.
    #
    # Now: fit on train, rank on validation, report the winner's test metrics.
    # best_on_test stays beside it, labelled as the oracle it is, so the
    # difference is a published number instead of an assumption.
    by_val = lambda s_: s_["mae_gdl_val"]                        # noqa: E731
    cands = ([out["exact"]] if out["exact"] else []) + fam

    out["learned"] = min(fam, key=by_val)
    out["selected"] = min(cands, key=by_val)
    out["best_on_test"] = min(cands, key=lambda s_: s_["mae_gdl"])
    out["selection_optimism_gdl"] = round(
        out["selected"]["mae_gdl"] - out["best_on_test"]["mae_gdl"], 6)
    out["selection_rule"] = (
        "fitted on the train block, ranked by validation MAE, reported on the "
        "test block. Test rows choose nothing. best_on_test is what selecting "
        "on test would have returned and is not the reported figure; "
        "selection_optimism_gdl is how much that shortcut would have added.")
    assert out["selection_optimism_gdl"] >= -1e-9, out["selection_optimism_gdl"]
    return out


def classify(df, cols, tr, te, target, strata, psu, multiclass=False):
    """Train the forest on one rung and score it properly."""
    X = df[cols].to_numpy()
    y = df[target].to_numpy()
    clf = forest()

    cv = cross_val_score(
        clf, X[tr], y[tr],
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1)

    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    yte = y[te]

    res = {
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_sd": float(cv.std(ddof=1)),
        "cv_folds": [float(v) for v in cv],
        # 18 of these run per pipeline invocation - nine rungs times two targets,
        # 90 forest fits, about 112 s - and nothing printed, plotted or tested
        # read a single one of them. They are kept rather than deleted because a
        # fold spread on the TRAINING block is the one honest answer to "is this
        # rung's accuracy a property of the data or of one partition", and
        # benchmark.py's repeated CV answers that for L1/L4/L6 only. Their
        # purpose is now stated and the spread is printed, so they are reported
        # numbers rather than a silent 112 s.
        "cv_note": ("5-fold accuracy on the TRAINING block only, as a seed- and "
                    "partition-sensitivity check on this rung. It is not the "
                    "reported performance - that is test_accuracy below, with a "
                    "design-based interval. benchmark.py repeats the same check "
                    "4 times over for three of these rungs."),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    }
    lo, hi, sd, degen = cluster_ci(yte, pred, strata, psu)
    res["test_accuracy_ci95"] = [lo, hi]
    res["test_accuracy_design_se"] = sd
    res["test_accuracy_ci_degenerate"] = degen
    res["test_accuracy_ci_method"] = (
        "Rao-Wu rescaled bootstrap replicate weights, t interval on "
        f"{dp.design_df(strata, psu)} df")

    if multiclass:
        res["macro_f1"] = float(f1_score(yte, pred, average="macro"))
        res["per_class_recall"] = [
            float(v) for v in recall_score(yte, pred, average=None,
                                           labels=sorted(set(y)), zero_division=0)]
        proba = None
    else:
        proba = clf.predict_proba(X[te])[:, 1]
        res["roc_auc"] = float(roc_auc_score(yte, proba))
        res["pr_auc"] = float(average_precision_score(yte, proba))
        res["pr_auc_no_skill"] = float(yte.mean())
        res["precision"] = float(precision_score(yte, pred, zero_division=0))
        res["recall"] = float(recall_score(yte, pred, zero_division=0))
        res["f1"] = float(f1_score(yte, pred, zero_division=0))
        rlo, rhi, _, rdegen = cluster_ci(
            yte, pred, strata, psu,
            stat=lambda a, b, sample_weight=None: recall_score(
                a, b, sample_weight=sample_weight, zero_division=0))
        res["recall_ci95"] = [rlo, rhi]
        res["recall_ci_degenerate"] = rdegen
        res["n_test_positives"] = int(yte.sum())

    res["feature_importance_mdi"] = dict(
        zip(cols, (float(v) for v in clf.feature_importances_)))
    res["feature_importance_note"] = (
        "MDI. Counts splits, so it inflates continuous high-cardinality "
        "features and flattens binary ones; see results/importance.json for the "
        "permutation ranking, which disagrees with this one.")
    return res, pred, proba, clf


def decompose(df, tr, te, strata, psu):
    """
    Separate the two sources of inflation in the submitted 99%.

    The manuscript made two independent choices that each inflate accuracy:
      (a) it kept MCH, so MCH*RBC/10 rebuilds hemoglobin
      (b) it applied a single 12.0 g/dL cut to men, women and children alike,
          which makes the label a function of hemoglobin ALONE

    Crossing feature set with label definition gives a 2x2. Two corrections to
    how the first version read it:

      The two effects are reported on the LIFT scale, accuracy minus that
      cell's own no-information rate. Changing the label definition changes the
      class balance, so part of the raw accuracy difference is just an easier
      majority class and is not attributable to leakage at all.

      The interaction is named and printed instead of being left as the gap
      between the two main effects and their sum. Removing the identity matters
      less once the label is the WHO one, and an additive summary hides that.
    """
    cells = {}
    for label, lname in [("Anemia_hb12", "sex-blind Hb<12"),
                         ("Anemia", "WHO age/sex/pregnancy")]:
        for rung in ("L1_paper_leaky", "L4_identity_free"):
            cols = dp.LADDER[rung]
            X, y = df[cols].to_numpy(), df[label].to_numpy()
            clf = forest().fit(X[tr], y[tr])
            pred, proba = clf.predict(X[te]), clf.predict_proba(X[te])[:, 1]
            yte = y[te]
            nir = float(max(yte.mean(), 1 - yte.mean()))
            acc = float(accuracy_score(yte, pred))
            cells[(rung, label)] = {
                "features": rung, "label": lname,
                "accuracy": acc,
                "no_information_rate": nir,
                "lift_points": round((acc - nir) * 100, 2),
                "pr_auc": float(average_precision_score(yte, proba)),
                "pr_auc_no_skill": float(yte.mean()),
                "recall": float(recall_score(yte, pred, zero_division=0)),
                "_pred": pred, "_y": yte,
            }

    def lift(rung, label):
        return cells[(rung, label)]["lift_points"]

    leaky, clean = "L1_paper_leaky", "L4_identity_free"
    old, who = "Anemia_hb12", "Anemia"
    ident_old = lift(leaky, old) - lift(clean, old)
    ident_who = lift(leaky, who) - lift(clean, who)
    label_leaky = lift(leaky, old) - lift(leaky, who)
    label_clean = lift(clean, old) - lift(clean, who)

    # paired tests where the label is held fixed, so the rows are the same
    tests = {}
    for label, tag in [(old, "sex-blind Hb<12"), (who, "WHO")]:
        a, b = cells[(leaky, label)], cells[(clean, label)]
        # (before, after) = (clean, leaky), so identity_effect is what HAVING
        # the identity buys - positive - matching the sign of ident_old/ident_who
        # on the lift scale above. _acc_diff returns after - before.
        tests[f"identity_effect_{tag}"] = paired_cluster_test(
            a["_y"], b["_pred"], a["_pred"], strata, psu)

    rows = [{k: v for k, v in c.items() if not k.startswith("_")}
            for c in cells.values()]

    # Holm across these two, and a statement of what they are NOT.
    #
    # These are the seventh and eighth paired tests in the file, and they are not
    # members of CONTRASTS: that family is six ladder-rung comparisons corrected
    # together, and these two are a different question asked on a different pair
    # of labels. Two tests reported with raw p-values beside a file whose whole
    # argument is that uncorrected families inflate significance is the practice
    # this paper criticises, and the earlier version did exactly that - printing
    # p = 5.8e-09 with no correction and no note saying it sat outside the
    # declared family. Correcting them here as their own family of two is the
    # honest reading: they were declared together, before the numbers, as the two
    # levels of one factor. Neither one moves across 0.05 either way.
    fam = list(tests)
    for k, adj in zip(fam, dp.holm([tests[k]["p_value"] for k in fam])):
        tests[k]["p_holm"] = adj
        tests[k]["significant_holm_05"] = bool(adj < 0.05)

    return {
        "cells": rows,
        "effects_on_lift_scale": {
            "identity_removal_under_old_label": round(ident_old, 2),
            "identity_removal_under_who_label": round(ident_who, 2),
            "label_correction_with_identity": round(label_leaky, 2),
            "label_correction_without_identity": round(label_clean, 2),
            "interaction": round(ident_old - ident_who, 2),
            "total_leaky_old_vs_clean_who": round(
                lift(leaky, old) - lift(clean, who), 2),
        },
        "tests": tests,
        "significance": {
            "family": fam,
            "correction": "Holm, over these two only",
            "note": ("these two tests are NOT members of the pre-declared "
                     "six-contrast family in `contrasts`; they are their own "
                     "family of two, declared as the two label levels of one "
                     "factor. p_holm is the value to read. They were previously "
                     "published with raw p-values and no such statement."),
        },
        "sign_convention":
            "Positive means the leaky choice scores higher. So "
            "identity_removal_* is what removing the identity costs, and "
            "label_correction_* is what correcting the label costs. The two "
            "main effects are each measured at the other factor's leaky level, "
            "so they do NOT sum to total_leaky_old_vs_clean_who: that would "
            "count the interaction twice. The two additive paths are "
            "identity_removal_under_old_label + label_correction_without_"
            "identity, and label_correction_with_identity + identity_removal_"
            "under_who_label. Both equal the total exactly (3.29 + 3.49 = "
            "5.30 + 1.48 = 6.78), which is what makes the decomposition a "
            "decomposition. What differs by the interaction is not the paths "
            "but the two estimates of each main effect: identity removal is "
            "3.29 points under the old label and 1.48 under the WHO one, and "
            "1.81 of the 3.29 is the interaction. This paragraph used to say "
            "the two paths differ by the interaction, which would require the "
            "interaction to be zero and it is not.",
        "note": "lift = accuracy - that cell's own no-information rate, because "
                "the label change also moves the class balance",
    }


def tracking(rungs):
    """
    Does classification accuracy track hemoglobin recoverability?

    Reported two ways, because the honest answer is 'within a fixed feature
    scope, yes; unconditionally, no'. L0 hands the model hemoglobin itself and
    still scores below L6, which has demographics instead -- so a claim that
    accuracy rises monotonically with recoverability over the whole ladder is
    false, and the first version of this study made it.
    """
    def spearman_p(x, yv):
        """
        Exact two-sided permutation p for Spearman's rho when n is small.

        scipy's default is a t approximation on n-2 df, and at n = 6 with a
        perfect monotone relation it returns exactly 0.0 - which is not a
        p-value, it is the approximation running out of resolution. The exact
        answer there is 2/6! = 0.002778: two of the 720 orderings are at least
        as extreme as the observed one. Every permutation is enumerated while
        n! is small enough to enumerate, and the method used is recorded.
        """
        rho0 = float(stats.spearmanr(x, yv).statistic)
        if not np.isfinite(rho0):
            # One of the two series is constant, so rho is 0/0. Left unguarded,
            # every `abs(rho) >= abs(nan)` comparison below is False, the exact
            # branch returns p = 0/720 = 0.0, and a NaN rho goes into a JSON
            # dump opened with allow_nan=False - so the run dies at write time
            # with a ValueError about the encoder rather than saying which
            # series was flat. It cannot happen on the current ladder (no two
            # rungs tie on either axis) and it is one added rung away.
            return None, 1.0, ("undefined: one series is constant, so Spearman's "
                               "rho is 0/0. Reported as p = 1 because no "
                               "monotone relation can be detected, not because "
                               "none exists")
        if len(x) > 8:
            return rho0, float(stats.spearmanr(x, yv).pvalue), \
                "t approximation on n-2 df"
        perms = list(itertools.permutations(range(len(yv))))
        extreme = sum(
            1 for pm in perms
            if abs(float(stats.spearmanr(x, [yv[i] for i in pm]).statistic))
            >= abs(rho0) - 1e-12)
        return rho0, extreme / len(perms), \
            f"exact, {extreme} of {len(perms)} permutations at least as extreme"

    def pack(names):
        if len(names) < 3:
            return None
        mae = np.array([rungs[k]["recoverability"]["selected"]["mae_gdl"]
                        for k in names])
        acc = np.array([rungs[k]["binary"]["test_accuracy"] for k in names])
        rho, p, how = spearman_p(mae, list(acc))
        return {"rungs": names, "spearman_rho": rho, "p_value": p,
                "p_method": how, "n": len(names)}

    def monotonicity_violations():
        """
        Selected-recoverability MAE cannot rise much when a rung gains features.

        The INFORMATION bound is monotone: if B's columns contain A's, nothing
        recoverable from A is unrecoverable from B. A fitted estimator is not,
        quite - extra near-useless regressors cost a little estimation variance,
        so a superset can come out infinitesimally worse. L1 vs L1+demographics
        is exactly that, about three parts in a million of a quantity the
        analyser itself reports to 0.1 g/dL.

        There is now a second, legitimate source: the winner is chosen on
        validation and scored on test, so a superset can pick a slightly
        different family member and land a little worse on test through
        selection noise alone. That is the price of not selecting on test, and
        it is the right price to pay.

        So the tolerance is 1e-3 g/dL: a hundredth of the reporting step, and
        still fifteen times smaller than the one real violation this check was
        written to catch - the log-model positivity gate made L6 look 0.0146
        g/dL worse than the L4 it contains. That was a truncated estimator
        family, not a property of the data, and it was invisible until here.
        Reported rather than asserted, because the right response to a genuine
        violation is to look at the family, not to crash.
        """
        tol = 1e-3
        bad = []
        for a in rungs:
            for b in rungs:
                if a == b:
                    continue
                if not set(rungs[a]["features"]) < set(rungs[b]["features"]):
                    continue
                ma = rungs[a]["recoverability"]["selected"]["mae_gdl"]
                mb = rungs[b]["recoverability"]["selected"]["mae_gdl"]
                if mb > ma + tol:
                    bad.append({"subset": a, "superset": b,
                                "subset_mae": ma, "superset_mae": mb,
                                "excess": round(mb - ma, 6)})
        return bad, tol

    no_demo = [n for n in rungs if not (set(dp.DEMO_FEATURES)
                                        & set(rungs[n]["features"]))]
    viol, tol = monotonicity_violations()
    return {"all_rungs": pack(list(rungs)),
            "fixed_scope_no_demographics": pack(no_demo),
            "monotonicity_violations": viol,
            "monotonicity_tolerance_gdl": tol,
            "monotonicity_note":
                f"Checked over every nested pair of rungs, tolerance {tol} g/dL "
                + ("- entries above are bugs in the recoverability family, not "
                   "results: more features cannot carry less information."
                   if viol else
                   "- none. Every rung recovers hemoglobin at least as well as "
                   "the rung its feature set contains, as it must. The "
                   "tolerance absorbs estimation variance from extra "
                   "near-useless regressors, which is why it is not zero."),
            "claim": "accuracy tracks recoverability when the feature scope is "
                     "held fixed; across scopes it does not, because "
                     "demographics buy accuracy without buying recoverability"}


def out_of_cycle(models):
    """
    Score every rung's fitted model on three NHANES cycles it never saw.

    Named out_of_cycle, not "external": these are other cycles of the SAME
    survey, with the same instrument and protocol, so "external validation" would
    overclaim. What changes between them is the households, the analyser
    calibration and the field period, which is enough to make them independent
    evidence and not enough to make them a different population.

    This is the part a single-cohort study cannot do. The train/test split puts
    people from the same PSU on both sides, so a test-set number is not
    independent evidence about a new sample; a different survey cycle, with
    different households and a different analyser calibration, is.

    Accuracy alone would be uninformative because prevalence differs by cycle,
    so lift over each cycle's own no-information rate is reported next to it.

    Takes only the fitted models. It used to take (df, tr) as well and read
    neither: the primary cohort is not involved in scoring a different cycle.
    """
    out = {}
    for cycle in CYCLES:
        if cycle == PRIMARY:
            continue
        rep, att = dp.load_cycle(cycle)
        per_rung = {}
        for rung, clf in models.items():
            cols = dp.LADDER[rung]
            X, y = rep[cols].to_numpy(), rep["Anemia"].to_numpy()
            pred = clf.predict(X)
            proba = clf.predict_proba(X)[:, 1]
            nir = float(max(y.mean(), 1 - y.mean()))
            acc = float(accuracy_score(y, pred))
            per_rung[rung] = {
                "accuracy": acc,
                "no_information_rate": nir,
                "lift_points": round((acc - nir) * 100, 2),
                "pr_auc": float(average_precision_score(y, proba)),
                "pr_auc_no_skill": float(y.mean()),
                "recall": float(recall_score(y, pred, zero_division=0)),
            }
        integ = att.get("integrity", {})
        out[cycle] = {"n": int(len(rep)),
                      "prevalence": float(rep["Anemia"].mean()),
                      "excluded_pct": att["excluded_pct"],
                      # The integrity screen runs on every cycle and nothing
                      # surfaced its verdict for the replication cycles, so
                      # 2013-2014's impossible row - SEQN 77520, MCHC 69.6 g/dL,
                      # internally consistent and externally unattainable - was
                      # adjudicated in a dataprep docstring and reported in no
                      # artefact. These models are scored on rows that include
                      # it. It is still not dropped; it is now named where the
                      # scores are.
                      "integrity": {
                          "values_outside_containment_bounds":
                              integ.get("values_outside_containment_bounds"),
                          "outlying_values": integ.get("outlying_values"),
                          "duplicate_seqn": integ.get("duplicate_seqn"),
                          "note": ("nothing is dropped on the basis of this "
                                   "screen in any cycle; the accuracy above is "
                                   "computed on every row, flagged ones "
                                   "included"),
                      },
                      "rungs": per_rung}
    return out


def train_size_sensitivity(df, tr, va, te, strata, psu):
    """
    Does the identity contrast survive if the models are given more data?

    Everything in this project is fitted on the 60% training split, so that the
    benchmark's validation split stays genuinely unused and every table in the
    paper is comparable. That rule was fixed for comparability, not for the
    p-value it produces -- and the way to show that is to report the contrast
    both ways. Fitting on train+validation (80%) is the other defensible choice,
    because the ladder tunes nothing and so has no need of a validation split.

    The two identity contrasts are recomputed under both training sets. If a
    conclusion moves, it is disclosed here rather than left for a reviewer to
    find by rerunning with a different split.
    """
    pairs = [("L1_paper_leaky", "L4_identity_free", "no demographics"),
             ("L1_paper_leaky_demo", "L6_identity_free_demo", "with demographics")]
    y = df["Anemia"].to_numpy()
    out = {}
    for tag, fit_idx in [("train_60pct", tr),
                         ("train_plus_val_80pct", np.sort(np.concatenate([tr, va])))]:
        preds = {}
        for rung in {r for p in pairs for r in p[:2]}:
            X = df[dp.LADDER[rung]].to_numpy()
            preds[rung] = forest().fit(X[fit_idx], y[fit_idx]).predict(X[te])
        out[tag] = {"n_fit": int(len(fit_idx)), "contrasts": {}}
        for a, b, what in pairs:
            t = paired_cluster_test(y[te], preds[a], preds[b], strata, psu)
            t["meaning"] = f"remove the identity, {what}"
            out[tag]["contrasts"][f"{a} vs {b}"] = t
    return out


def complete_case_sensitivity(df, tr, te, preds, strata, psu):
    """
    What does sharing one cohort across all nine rungs actually cost?

    dataprep.ALL_FEATURE_COLS requires completeness on the UNION of every rung's
    features, so a rung that uses only {RBC, MCV, RDW} is held to L5's platelet
    and white-cell channels too. That is the right choice - rungs fitted on
    different rows are not comparable, and comparing rungs is the whole study -
    but it looks like it should cost rows, and this file used to say so: "costs
    1,616 rows that L6 could otherwise have kept".

    That was wrong, and this function is what shows it. In the primary cycle the
    CBC panel is missing PERSON BY PERSON, not analyte by analyte: of 13,772
    merged records, 12,156 have all nine values and 1,616 have none of them - and
    not just none of the nine, none of the 21 CBC columns in the file at all. The
    missingness is one blood draw that did not happen, so no narrower feature set
    can recover a single row of it. Every rung's own maximal complete-case cohort
    is the shared cohort, exactly.

    So the bill is zero, and it is zero for a reason rather than by luck. Three
    things follow, and all three are reported:

      1. The union requirement is free, so the comparability it buys costs
         nothing and needs no defending.
      2. The complete-case attrition documented in `cohort.attrition_profile` -
         younger, SMD -0.873 - is a property of the SURVEY, not of this feature
         list. No rung choice would reduce it. That is a stronger statement of
         the limitation than the one it replaces.
      3. Nothing here can be fixed by dropping L5.

    The refit arm below is written anyway, and on the primary cycle it never
    executes, because no rung there gains a row. Say that plainly rather than
    implying otherwise: an earlier version of this docstring said the arm "runs
    whenever a rung DOES gain rows, which happens in one replication cycle
    (2011-2012 has a single record with every value but platelets)", which reads
    as though the arm runs on that cycle. It does not. The replication cycles are
    COUNTED here and refitted nowhere - the models in this file are fitted on the
    primary cycle only, and out_of_cycle scores them elsewhere. So the arm is
    dormant code with a live purpose: it is the check that would fire if a future
    cycle became the primary one, and `rungs_with_extended_fit` is empty rather
    than absent so a reader can see that it did not fire rather than assume it
    did.

    Where it does run, the extra rows are added to the TRAINING block only and
    the rung is rescored on the identical shared test rows, so the comparison is
    paired and no forfeited row can leak into an evaluation: a row absent from
    the shared cohort is by construction absent from the shared test block. What
    that arm cannot answer is whether a rung would do as well on the forfeited
    people themselves - they have no hemoglobin, so they have no label, so no
    accuracy exists for them under any feature set. That is a ceiling on the
    question, not on this method.
    """
    raw, _, counts = dp.merged_raw()
    shared = set(df["SEQN"].to_numpy().tolist())
    y = df["Anemia"].to_numpy()
    yte = y[te]
    cbc_cols = dp.ALL_FEATURE_COLS

    # The mechanism, measured rather than asserted: how many of the nine are
    # missing per row, and whether the all-nine rows have ANY CBC value.
    n_missing = raw[cbc_cols].isna().sum(axis=1).to_numpy()
    vals, cnts = np.unique(n_missing, return_counts=True)
    lb_cols = [c for c in raw.columns if c.startswith("LB")]
    no_panel = raw.loc[n_missing == len(cbc_cols)]
    out = {
        "question": ("does requiring completeness on the union of all ladder "
                     "features cost any rung rows it could have used?"),
        "answer_primary_cycle": None,          # filled in below, from the counts
        "merged_records": int(len(raw)),
        "shared_cohort_n": int(len(df)),
        "missing_columns_per_row": {int(k): int(v) for k, v in zip(vals, cnts)},
        "missingness_pattern": (
            "all-or-nothing at the person level: every row is complete on all "
            f"{len(cbc_cols)} CBC columns or missing all {len(cbc_cols)}"
            if set(vals.tolist()) <= {0, len(cbc_cols)} else
            "not all-or-nothing; some rows are partially observed"),
        "rows_with_no_panel": int(len(no_panel)),
        "cbc_columns_in_file": len(lb_cols),
        "rows_with_no_panel_having_any_cbc_value": int(
            no_panel[lb_cols].notna().any(axis=1).sum()),
        "mechanism": ("a missing CBC is a blood draw that did not happen, not an "
                      "analyte that failed, so no narrower feature set recovers "
                      "any of it"),
        "rungs": {},
    }

    gained = []
    for rung, cols in dp.LADDER.items():
        own, need = dp.cohort_on(cols, raw=raw)
        extra = own.loc[~own["SEQN"].isin(shared)]
        rec = {"required_columns": list(need),
               "n_complete_on_own_columns": int(len(own)),
               "rows_forgone_to_shared_cohort": int(len(extra)),
               "shared_cohort_is_maximal": bool(len(extra) == 0)}
        if len(extra):
            gained.append(rung)
            keep = list(cols) + ["Anemia"]
            fit = pd.concat([df.iloc[tr][keep], extra[keep]], ignore_index=True)
            assert not (set(extra["SEQN"].to_numpy().tolist())
                        & set(df.iloc[te]["SEQN"].to_numpy().tolist())), \
                "a forgone row cannot be in the shared test block"
            p = forest().fit(fit[list(cols)].to_numpy(),
                             fit["Anemia"].to_numpy()).predict(
                                 df[cols].to_numpy()[te])
            t = paired_cluster_test(yte, preds[rung], p, strata, psu)
            t["meaning"] = (f"add the {len(extra)} forgone row(s) to the training "
                            "block; same test rows")
            rec["n_fit_shared"] = int(len(tr))
            rec["n_fit_extended"] = int(len(fit))
            rec["extended_fit_test"] = t
        else:
            rec["extended_fit_test"] = None
            rec["why_no_test"] = ("nothing to add: this rung's own complete-case "
                                  "cohort is the shared cohort")
        out["rungs"][rung] = rec

    forgone = {r: v["rows_forgone_to_shared_cohort"]
               for r, v in out["rungs"].items()}
    out["max_rows_forgone_by_any_rung"] = int(max(forgone.values()))
    out["answer_primary_cycle"] = (
        "no: every rung's own complete-case cohort is the shared cohort"
        if max(forgone.values()) == 0 else
        f"yes, for {sorted(gained)}")
    out["rungs_with_extended_fit"] = sorted(gained)

    # The same count on the three replication cycles. Cheap - one merge each,
    # then a dropna per rung - and it is the only way to know whether the
    # all-or-nothing pattern is a fact about this survey or about this cycle.
    out["replication_cycles"] = {}
    for cycle in CYCLES:
        if cycle == PRIMARY:
            continue
        r2, _, _ = dp.merged_raw(cycle)
        base = int(len(r2.dropna(subset=cbc_cols)))
        per = {}
        for rung, cols in dp.LADDER.items():
            need = dp.required_columns(cols)
            per[rung] = int(len(r2.dropna(subset=need))) - base
        k = r2[cbc_cols].isna().sum(axis=1).to_numpy()
        v2, c2 = np.unique(k, return_counts=True)
        out["replication_cycles"][cycle] = {
            "merged_records": int(len(r2)),
            "shared_cohort_n": base,
            "missing_columns_per_row": {int(a): int(b) for a, b in zip(v2, c2)},
            "rows_forgone_by_rung": per,
            "max_rows_forgone": int(max(per.values())),
        }
    out["replication_note"] = (
        "2015-2016 has 30 records carrying MCHC and nothing else. They are "
        "unusable at every rung, not just the wide ones, because they have no "
        "hemoglobin and therefore no label. 2011-2012 has one record complete "
        "except for platelets, which is the only row in four cycles that the "
        "union requirement actually costs anyone.")
    return out


def cluster_holdout(df, seed=SEED):
    """
    The check dp.split3's docstring promises: does cluster optimism cancel?

    dp.split3 is i.i.d. over ROWS, so all 49 PSUs appear in both the training
    and the test block. Test people therefore come from neighbourhoods the model
    has already seen, and the absolute accuracies carry whatever optimism that
    buys. The project's defence is that its claims are paired DIFFERENCES on
    identical rows, where optimism common to both arms cancels. That was an
    assertion. This measures it.

    Two folds, split on PSUs rather than rows. Within each stratum the PSUs are
    shuffled with a fixed seed and dealt alternately, so every stratum is
    represented in both folds and no PSU is ever in both the training and the
    test side of the same fold. Each fold refits the six rungs the contrast
    family names - not all nine; L2, L3 and L5 appear in no pre-declared
    contrast - and recomputes the six contrasts on its held-out PSUs. This
    paragraph used to say "all seven rungs", which is neither number.

    Two numbers come out of it.

      ABSOLUTE   the drop in each rung's accuracy from the row-wise split to
                 genuinely unseen clusters. This is the optimism, and it is real.
      PAIRED     the same six contrasts under both regimes. If the cancellation
                 claim is true these agree; if it is false they do not, and the
                 paper would have to be rewritten around it.

    One confound is disclosed rather than removed. A fold trains on about half
    the cohort where dp.split3 trains on 60%, so the accuracy gap below mixes
    cluster optimism with a training-size effect and is therefore an UPPER bound
    on the optimism alone. train_size_sensitivity measures the direction of that
    second effect on the same contrasts. Equalising the two would mean either
    discarding PSUs or fitting on fewer rows than the rest of the project, and
    both cost more than the confound does: the quantity that matters here is
    whether the paired contrasts survive, and an upper bound on the nuisance term
    is the conservative side to err on.

    No design-based interval is reported here, and that is not an omission.
    Every fold's held-out set is one PSU per stratum for most strata, so there
    is no within-stratum replication left to estimate a variance from - the
    Rao-Wu scheme has nothing to resample. An i.i.d. row bootstrap SE is given
    instead, as a scale for reading the fold-to-fold spread, and labelled as
    what it is. The point of this function is the point estimates.
    """
    y = df["Anemia"].to_numpy()
    strata_all, psu_all, _ = dp.design_arrays(df)
    rungs_used = sorted({r for c in CONTRASTS for r in c[:2]})

    # Deal each stratum's PSUs alternately into two folds, so both folds cover
    # all 24 strata and neither shares a PSU with its own training side.
    rng = np.random.default_rng(seed)
    fold_of_psu = {}
    for h in np.unique(strata_all):
        ps = np.unique(psu_all[strata_all == h])
        order = rng.permutation(len(ps))
        for j, k in enumerate(order):
            fold_of_psu[(int(h), int(ps[k]))] = j % 2
    key = np.array([fold_of_psu[(int(a), int(b))]
                    for a, b in zip(strata_all, psu_all)])

    folds = []
    for f in (0, 1):
        te = np.flatnonzero(key == f)
        tr = np.flatnonzero(key != f)
        preds, accs = {}, {}
        for rung in rungs_used:
            X = df[dp.LADDER[rung]].to_numpy()
            preds[rung] = forest().fit(X[tr], y[tr]).predict(X[te])
            accs[rung] = round(float(accuracy_score(y[te], preds[rung])), 6)
        con = {}
        for a, b_, what in CONTRASTS:
            rows = paired_row_test(y[te], preds[a], preds[b_])
            con[f"{a} vs {b_}"] = {
                "meaning": what,
                "delta_accuracy": round(_acc_diff(y[te], preds[a], preds[b_]), 6),
                "iid_row_bootstrap_se": round(rows["se"], 6),
                "se_note": "i.i.d. over rows. The design-based estimator has no "
                           "within-stratum replication left in a held-out fold "
                           "of one PSU per stratum, so it cannot be used here.",
            }
        # PSU identity is the (stratum, PSU) PAIR, because SDMVPSU is numbered
        # 1..3 WITHIN a stratum and means nothing on its own. This used to be
        # counted as `psu + 100 * strata`, which is injective only while PSU
        # codes stay below 100 - true in every NHANES cycle so far and not a
        # property anything here checks. And psu_overlap_with_train was the
        # literal 0: the file asserted the scheme's defining guarantee instead of
        # measuring it, in the one function whose entire purpose is to measure a
        # thing the project had been asserting.
        cell = set(zip(strata_all.tolist(), psu_all.tolist()))
        tr_cells = set(zip(strata_all[tr].tolist(), psu_all[tr].tolist()))
        te_cells = set(zip(strata_all[te].tolist(), psu_all[te].tolist()))
        assert not (tr_cells & te_cells), sorted(tr_cells & te_cells)
        folds.append({
            "fold": f,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_train_psu": len(tr_cells),
            "n_test_psu": len(te_cells),
            "n_psu_total": len(cell),
            "psu_overlap_with_train": len(tr_cells & te_cells),
            "psu_identity": "the (SDMVSTRA, SDMVPSU) pair; SDMVPSU is numbered "
                            "within stratum and is not unique on its own",
            "test_prevalence": round(float(y[te].mean()), 6),
            "accuracy_by_rung": accs,
            "contrasts": con,
        })

    mean_con = {}
    for k in folds[0]["contrasts"]:
        vals = [f["contrasts"][k]["delta_accuracy"] for f in folds]
        mean_con[k] = {
            "meaning": folds[0]["contrasts"][k]["meaning"],
            "delta_accuracy_mean": round(float(np.mean(vals)), 6),
            "delta_accuracy_by_fold": vals,
            "fold_spread": round(float(abs(vals[0] - vals[1])), 6),
        }
    mean_acc = {r: round(float(np.mean([f["accuracy_by_rung"][r]
                                       for f in folds])), 6)
                for r in rungs_used}
    return {"scheme": "PSUs dealt alternately within stratum into 2 folds; "
                      "train and test share no PSU",
            "seed": seed,
            "folds": folds,
            "mean_accuracy_by_rung": mean_acc,
            "mean_contrasts": mean_con}


def main():
    df, att = dp.load_cohort()
    tr, va, te = dp.split3(df)
    strata, psu, _ = dp.design_arrays(df, te)

    y = df["Anemia"].to_numpy()
    # Baselines are computed on the TEST set, because that is what every model
    # number in this file is scored on. Using the whole cohort instead gives a
    # slightly different figure and puts two baselines in one paper.
    yte_bin = y[te]
    maj_bin = max(yte_bin.mean(), 1 - yte_bin.mean())
    sev = df["Severity"].to_numpy()[te]
    maj_sev = np.bincount(sev).max() / len(sev)

    print(f"cohort {att['n_analysis']:,}   train {len(tr):,}   "
          f"val {len(va):,} (held out, unused here)   test {len(te):,}")
    print(f"test no-information rate: binary {maj_bin*100:.2f}%   "
          f"severity {maj_sev*100:.2f}%")
    print(f"test PSU clusters: {len(set(zip(strata, psu)))} in "
          f"{len(set(strata))} strata  ({len(te):,} rows)\n")

    out = {
        "cohort": att,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "fit_rule": "every model in this file is fitted on the 60% training "
                    "split only, so that benchmark.py's validation split stays "
                    "unused and every number in the paper is comparable; see "
                    "train_size_sensitivity for the contrast under an 80% fit",
        "n_test_clusters": int(len(set(zip(strata, psu)))),
        "majority_baseline_binary": float(maj_bin),
        "majority_baseline_severity": float(maj_sev),
        "pr_auc_no_skill": float(yte_bin.mean()),
        "variance_estimator": (
            f"Rao-Wu rescaled bootstrap, {N_REP} replicates, "
            f"{dp.design_df(strata, psu)} df"),
        "estimand": (
            "UNWEIGHTED within-sample accuracy contrasts, with stratum- and "
            "PSU-aware uncertainty. The Rao-Wu replicate weights are built on "
            "unit base weights (dataprep.rao_wu_weights(base=None)), so every "
            "interval and p-value in this file answers 'does rung A beat rung "
            "B on these rows, once clustering is accounted for', NOT 'what "
            "would either score in the US population'. Weighting these "
            "contrasts would change the question. The population estimates are "
            "in results/weighted.json, which passes MECWT as the base weight. "
            "See variance_scales below for the measured cost of the design on "
            "these differences: 0.90x the i.i.d.-row STANDARD ERROR. The design "
            "effect of 4.00 that belongs to the weighted prevalence is a "
            "VARIANCE ratio, so on the standard-error scale it is 2.00 - this "
            "sentence used to compare 0.90 against 4.00 directly, which "
            "overstates the difference between the two by a factor of two."),
        "rungs": {},
    }

    preds, models = {}, {}
    for rung, cols in dp.LADDER.items():
        print(f"--- {rung}  {cols}")
        b, pred, proba, clf = classify(df, cols, tr, te, "Anemia", strata, psu)
        s, _, _, _ = classify(df, cols, tr, te, "Severity", strata, psu,
                              multiclass=True)
        r = {"features": list(cols),
             "recoverability": hb_recoverability(df, cols, tr, va, te),
             "binary": b, "severity": s}
        out["rungs"][rung] = r
        preds[rung] = pred
        models[rung] = clf

        ex = r["recoverability"]["exact"]
        be = r["recoverability"]["selected"]
        print(f"    Hb recoverable: exact "
              f"{'MAE %.4f g/dL via %s' % (ex['mae_gdl'], ex['route']) if ex else 'no identity'}")
        print(f"                    selected on validation: {be['estimator']}, "
              f"test MAE {be['mae_gdl']:.4f} g/dL, label agreement "
              f"{be['label_agreement']*100:.2f}%  "
              f"(selecting on test would have given "
              f"{r['recoverability']['best_on_test']['mae_gdl']:.4f})")
        print(f"    binary  acc {b['test_accuracy']:.4f} "
              f"[{b['test_accuracy_ci95'][0]:.4f}, {b['test_accuracy_ci95'][1]:.4f}]  "
              f"PR-AUC {b['pr_auc']:.4f}  recall {b['recall']:.4f} "
              f"[{b['recall_ci95'][0]:.3f}, {b['recall_ci95'][1]:.3f}]")

    # ------------------------------------------------------------ the table --
    print("\n\nTABLE: leakage ladder  (CIs are design-based: Rao-Wu replicate "
          "weights on 25 df)")
    print(f"{'rung':<24} {'Hb MAE':>8} {'via':>14} {'agree':>7} | "
          f"{'test acc':>8} {'95% CI':>15} {'PR-AUC':>7} {'recall':>7} | {'sev acc':>8}")
    print("-" * 124)
    for rung, r in out["rungs"].items():
        be, b, s = r["recoverability"]["selected"], r["binary"], r["severity"]
        ci = f"[{b['test_accuracy_ci95'][0]:.3f},{b['test_accuracy_ci95'][1]:.3f}]"
        print(f"{rung:<24} {be['mae_gdl']:8.4f} {be['estimator'][:14]:>14} "
              f"{be['label_agreement']*100:6.2f}% | "
              f"{b['test_accuracy']:8.4f} {ci:>15} {b['pr_auc']:7.4f} "
              f"{b['recall']:7.4f} | {s['test_accuracy']:8.4f}")
    print("-" * 124)
    print(f"{'no-information rate':<24} {'-':>8} {'-':>14} {'-':>7} | "
          f"{maj_bin:8.4f} {'-':>15} {yte_bin.mean():7.4f} {0.0:7.4f} | "
          f"{maj_sev:8.4f}")
    print(f"PR-AUC's no-skill value is the prevalence, {yte_bin.mean():.4f}, not "
          f"the no-information rate.")
    print("Hb MAE is the estimator chosen by validation MAE and then scored "
          "once on test; the test block selects nothing. ablation.json also "
          "records best_on_test, the oracle that selecting on test would give, "
          "and selection_optimism_gdl, the difference.")
    print("training-block 5-fold spread, for partition sensitivity (not the "
          "reported performance):")
    for rung, r in out["rungs"].items():
        b, s = r["binary"], r["severity"]
        print(f"  {rung:<24} binary {b['cv_accuracy_mean']:.4f} "
              f"+/-{b['cv_accuracy_sd']:.4f}   severity "
              f"{s['cv_accuracy_mean']:.4f} +/-{s['cv_accuracy_sd']:.4f}")

    # ------------------------------------------- the pre-declared contrasts --
    print("\n\nPRE-DECLARED CONTRASTS  (6 tests, Holm-corrected together)")
    design, naive, rows_iid = [], [], []
    for a, b_, _ in CONTRASTS:
        design.append(paired_cluster_test(yte_bin, preds[a], preds[b_],
                                         strata, psu))
        naive.append(paired_naive_cluster_test(yte_bin, preds[a], preds[b_],
                                               strata, psu))
        rows_iid.append(paired_row_test(yte_bin, preds[a], preds[b_]))
    adj = holm([t["p_value"] for t in design])
    out["contrasts"] = []
    print(f"  {'contrast':<44}{'delta':>8}{'Rao-Wu 95% CI':>19}"
          f"{'p':>7}{'Holm':>7} | {'SE row':>8}{'SE naive':>9}{'SE R-W':>8}")
    for (a, b_, what), t, nt, rt, pa in zip(CONTRASTS, design, naive,
                                            rows_iid, adj):
        t["contrast"] = f"{a} vs {b_}"
        t["meaning"] = what
        t["p_holm"] = pa
        t["significant_holm_05"] = bool(pa < 0.05)
        t["naive_cluster_bootstrap"] = nt
        t["iid_row_bootstrap"] = rt
        t["se_ratio_design_over_row"] = round(t["se"] / max(rt["se"], 1e-12), 2)
        t["se_ratio_naive_over_row"] = round(nt["se"] / max(rt["se"], 1e-12), 2)
        out["contrasts"].append(t)
        ci = f"[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}]"
        mark = "*" if pa < 0.05 else " "
        print(f"  {what:<44}{t['delta_accuracy']:+8.4f}{ci:>19}"
              f"{t['p_value']:>7.3f}{pa:>7.3f}{mark}|{rt['se']:>8.5f}"
              f"{nt['se']:>9.5f}{t['se']:>8.5f}")

    r_des = [t["se_ratio_design_over_row"] for t in out["contrasts"]]
    r_nai = [t["se_ratio_naive_over_row"] for t in out["contrasts"]]

    # The worked example in the sign-convention note is read out of the result
    # rather than typed into it. It used to say "-0.0148" in two places, which is
    # a number that moves whenever the seed, the forest or the split does, in a
    # field whose job is to tell a reader how to read the sign of the number
    # beside it.
    _ex = next((t for t in out["contrasts"]
                if t["contrast"] == "L1_paper_leaky vs L4_identity_free"), None)
    _d = _ex["delta_accuracy"] if _ex else None
    out["contrasts_sign_convention"] = (
        "accuracy(after) - accuracy(before), where 'before' and 'after' are the "
        "two rungs named in each contrast string, in that order. A negative "
        "delta therefore means the second rung scores lower - e.g. "
        f"'L1_paper_leaky vs L4_identity_free' at {_d:+.4f} means removing the "
        f"identity costs {abs(_d) * 100:.2f} accuracy points. NOTE the "
        "decomposition block signs the SAME quantities the other way round "
        "(positive means the leaky choice scores higher, because there the "
        "numbers are named as costs and benefits rather than as transitions), so "
        f"identity_effect_WHO is {-_d:+.4f} there and {_d:+.4f} here. "
        "Magnitudes, SEs and p-values are identical; only the sign convention "
        "differs, and each block states its own."
        if _ex else
        "accuracy(after) - accuracy(before), in the order the two rungs are "
        "named in each contrast string. The decomposition block signs the same "
        "quantities the other way round and says so.")

    out["variance_scales"] = {
        "replicates": N_REP,
        "design_df": dp.design_df(strata, psu),
        "median_se_ratio_raowu_over_row": float(np.median(r_des)),
        "median_se_ratio_naive_cluster_over_row": float(np.median(r_nai)),
        "se_ratio_raowu_over_row_range": [float(min(r_des)), float(max(r_des))],
        "scale_note": (
            "These are ratios of STANDARD ERRORS. weighted.json's design effect "
            "of 4.00 is a ratio of VARIANCES, so the two are not on one scale: "
            "4.00 on the variance scale is 2.00 on this one. The comparison to "
            "make is 0.90x against 2.00x, not against 4.00x, and stating it the "
            "other way overstated the contrast by a factor of two - in the "
            "direction that flatters the file's own point. The per-contrast "
            "range is also reported, because the median hides it: across the six "
            "contrasts the Rao-Wu SE runs from below to slightly above the "
            "i.i.d.-row SE, so 'the design penalty is modest' is a statement "
            "about the middle of a spread that crosses 1.0, not about a constant "
            "factor."),
        "note": "Three variance scales for the same six contrasts. Rao-Wu is "
                "the one the paper reports. The naive n_h-of-n_h cluster "
                "bootstrap has variance expectation (n_h-1)/n_h of the truth, "
                "which at 2 PSUs per stratum is half, so it comes out NARROWER "
                "than the i.i.d. row bootstrap and would be the wrong "
                "correction to apply.",
    }
    print(f"  * = survives Holm at 0.05.  p is a t test on "
          f"{dp.design_df(strata, psu)} df, not a bootstrap percentile.")
    print(f"  standard errors, median relative to the i.i.d. row bootstrap: "
          f"naive cluster {np.median(r_nai):.2f}x, Rao-Wu "
          f"{np.median(r_des):.2f}x  (Rao-Wu range {min(r_des):.2f}x to "
          f"{max(r_des):.2f}x)")
    print("  the naive cluster bootstrap draws n_h of n_h PSUs, so at 2 PSUs per")
    print("  stratum its variance expectation is (n_h-1)/n_h = 1/2 of the truth;")
    print("  it therefore comes out narrower than the i.i.d. interval and is not")
    print("  the correction it looks like. Rao-Wu rescaling is the valid scheme,")
    print("  and the honest finding is that for an unweighted PAIRED accuracy")
    print("  difference the design penalty is modest. Compare like with like: the")
    print("  design effect of 4.00 in weighted.py is a VARIANCE ratio, i.e. 2.00")
    print("  on the standard-error scale used above, and it belongs to a weighted")
    print("  PREVALENCE where unequal weights dominate. It does not transfer to")
    print("  this contrast.")

    surv = [t for t in out["contrasts"] if t["significant_holm_05"]]
    fell = [t for t in out["contrasts"] if not t["significant_holm_05"]]
    out["surviving_contrasts"] = [t["contrast"] for t in surv]
    out["null_contrasts"] = [t["contrast"] for t in fell]
    # The verdict string is built AFTER train_size_sensitivity runs, below.
    # It used to be written here and it named which contrast was split-sensitive
    # in hardcoded prose, inside a field whose whole purpose is to be computed -
    # so if the flip ever moved to a different contrast, or stopped happening,
    # the JSON would have gone on asserting the old one.
    print(f"\n  {len(surv)} of {len(out['contrasts'])} contrasts survive Holm:")
    for t in surv:
        print(f"    + {t['meaning']:<42}{t['delta_accuracy']:+.4f}  "
              f"Holm p {t['p_holm']:.4f}")
    for t in fell:
        print(f"    - {t['meaning']:<42}{t['delta_accuracy']:+.4f}  "
              f"Holm p {t['p_holm']:.4f}  (reported as null)")
    swap = next((t for t in out["contrasts"]
                 if t["contrast"] == "L0_hgb_direct vs L1_paper_leaky"), None)
    if swap is not None and not swap["significant_holm_05"]:
        print("  the null in that list is the informative one: replacing "
              "hemoglobin with")
        print(f"  the MCH*RBC/10 route that reconstructs it moves accuracy by "
              f"{swap['delta_accuracy']:+.4f}")
        print("  (Holm p {:.3f}), so a feature list without hemoglobin in it is "
              "not a".format(swap["p_holm"]))
        print("  feature list without hemoglobin in it.")

    # --------------------------------------------- training-size sensitivity --
    out["train_size_sensitivity"] = train_size_sensitivity(
        df, tr, va, te, strata, psu)
    print("\n  SENSITIVITY TO THE TRAINING SPLIT (same test rows, same design)")
    print(f"    {'contrast':<40}{'fit on':>20}{'delta':>9}{'p':>8}")
    flips = []
    short = {"train_60pct": "train only",
             "train_plus_val_80pct": "train+val"}
    for tag, blk in out["train_size_sensitivity"].items():
        for key, t in blk["contrasts"].items():
            where = f"{short.get(tag, tag)} {blk['n_fit']:,}"
            print(f"    {t['meaning']:<40}{where:>20}"
                  f"{t['delta_accuracy']:+9.4f}{t['p_value']:>8.3f}")
    a = out["train_size_sensitivity"]["train_60pct"]["contrasts"]
    b = out["train_size_sensitivity"]["train_plus_val_80pct"]["contrasts"]

    # The 60% arm is not a second result. Same seed, same forest, same training
    # rows, same test rows as the pre-declared contrast of the same name, so the
    # predictions are identical and so is the test. It was being printed and
    # stored a second time with no cross-reference and without the Holm
    # adjustment its twin carries, which reads as two independent confirmations
    # of one number. Each 60% entry now points at the contrast it duplicates and
    # carries that contrast's corrected p; the informative arm is the 80% one.
    by_contrast = {c["contrast"]: c for c in out["contrasts"]}
    for key, t60 in a.items():
        t60["duplicate_of"] = key if key in by_contrast else None
        if t60["duplicate_of"] is not None:
            twin = by_contrast[key]
            t60["p_holm_from_declared_family"] = twin["p_holm"]
            t60["note"] = (
                "Identical to the pre-declared contrast of the same name - same "
                "fit, same rows. Kept only as the baseline arm of this "
                "sensitivity check; it is not a second test.")
            assert abs(t60["delta_accuracy"] - twin["delta_accuracy"]) < 1e-12, \
                f"{key}: the 60% arm should reproduce the declared contrast"

    # The flip test compares two training sizes, so it uses the UNCORRECTED p in
    # both arms: that is the only like-for-like available, because the 80% arm
    # was never part of the pre-declared family and so has no Holm adjustment.
    # The headline verdict elsewhere is Holm-corrected, and saying so here stops
    # the two thresholds being read as the same one.
    for key in a:
        if (a[key]["p_value"] < 0.05) != (b[key]["p_value"] < 0.05):
            flips.append(key)
    out["train_size_sensitivity"]["conclusions_that_flip_at_05"] = flips
    out["train_size_sensitivity"]["flip_criterion"] = (
        "Uncorrected p < 0.05 in both arms. The 80% arm is outside the "
        "pre-declared family, so it has no Holm-adjusted p to compare against; "
        "an uncorrected-vs-uncorrected comparison is the only like-for-like "
        "one. The reported significance of the contrasts themselves is "
        "Holm-corrected and is a stricter bar than this.")
    holm_flips = [k for k in a
                  if a[k].get("p_holm_from_declared_family") is not None
                  and ((a[k]["p_holm_from_declared_family"] < 0.05)
                       != (b[k]["p_value"] < 0.05))]
    out["train_size_sensitivity"]["flips_if_60pct_arm_uses_holm"] = holm_flips
    if flips:
        print(f"    {len(flips)} conclusion(s) change with the training split: "
              + "; ".join(flips))
        print("    this is disclosed rather than resolved by picking the split "
              "that suits;")
        print("    the 60% fit is the reported one because it is the rule the "
              "benchmark needs.")
    else:
        print("    no conclusion changes with the training split")

    ident = [t for t in out["contrasts"] if t["meaning"].startswith("remove the")]
    both_cost = all(t["delta_accuracy"] < 0 for t in ident)
    out["contrast_verdict"] = (
        f"{len(surv)} of {len(out['contrasts'])} pre-declared contrasts survive "
        f"Holm under a design-based variance. Surviving: "
        + ("; ".join(t["contrast"] for t in surv) if surv else "none")
        + ". Reported as null: "
        + ("; ".join(t["contrast"] for t in fell) if fell else "none")
        + ". Removing the identity costs accuracy in "
        + ("both feature scopes" if both_cost
           else f"{sum(1 for t in ident if t['delta_accuracy'] < 0)} of "
                f"{len(ident)} feature scopes")
        + ", and "
        + (f"the split-sensitive result is {'; '.join(flips)} -- see "
           f"train_size_sensitivity, which reports it both ways instead of "
           f"choosing" if flips else
           "no conclusion in this file depends on how much training data the "
           "models are given; train_size_sensitivity reports both fits and "
           "they agree")
        + ".")
    print(f"\n  verdict: {out['contrast_verdict']}")

    # --------------------------------------- rung-specific complete-case check --
    out["complete_case_sensitivity"] = complete_case_sensitivity(
        df, tr, te, preds, strata, psu)
    cc = out["complete_case_sensitivity"]
    print("\n  RUNG-SPECIFIC COMPLETE-CASE CHECK (what the shared cohort costs)")
    print(f"    {'rung':<24}{'own columns':>13}{'own cohort n':>14}"
          f"{'rows forgone':>14}")
    for rung, rec in cc["rungs"].items():
        print(f"    {rung:<24}{len(rec['required_columns']):>13}"
              f"{rec['n_complete_on_own_columns']:>14,}"
              f"{rec['rows_forgone_to_shared_cohort']:>14,}")
    print(f"    {cc['missingness_pattern']}")
    print(f"    {cc['rows_with_no_panel']:,} rows have no CBC panel, and "
          f"{cc['rows_with_no_panel_having_any_cbc_value']} of them carry any of "
          f"the {cc['cbc_columns_in_file']} CBC columns in the file")
    print(f"    -> {cc['answer_primary_cycle']}")
    print("    so the complete-case attrition is a property of the survey, not "
          "of this feature list:")
    print("    no rung choice reduces it, and dropping L5 would not buy a row.")
    for cycle, blk in cc["replication_cycles"].items():
        print(f"    {cycle}: shared cohort {blk['shared_cohort_n']:,}, "
              f"most any rung forgoes {blk['max_rows_forgone']}")
    for rung, rec in cc["rungs"].items():
        t = rec["extended_fit_test"]
        if t is not None:
            print(f"    {rung}: refitted with {rec['n_fit_extended'] - rec['n_fit_shared']} "
                  f"extra training row(s), delta {t['delta_accuracy']:+.4f} "
                  f"(p {t['p_value']:.3f})")
    if not cc["rungs_with_extended_fit"]:
        print("    no rung was refitted: on this cycle no rung forgoes a row, so "
              "the refit arm has nothing to add")

    # ------------------------------------------------- cluster-held-out check --
    out["cluster_holdout"] = cluster_holdout(df)
    ch = out["cluster_holdout"]
    print("\n  CLUSTER-HELD-OUT CHECK (train and test share no PSU)")
    print(f"    {ch['scheme']}")
    print(f"    {'rung':<24}{'row split':>11}{'held-out PSU':>14}{'optimism':>10}")
    for rung, a_ch in ch["mean_accuracy_by_rung"].items():
        a_row = out["rungs"][rung]["binary"]["test_accuracy"]
        print(f"    {rung:<24}{a_row:11.4f}{a_ch:14.4f}{a_row - a_ch:+10.4f}")
    opt = [out["rungs"][r]["binary"]["test_accuracy"] - a
           for r, a in ch["mean_accuracy_by_rung"].items()]
    ch["absolute_optimism_mean"] = round(float(np.mean(opt)), 6)
    ch["absolute_optimism_max"] = round(float(np.max(opt)), 6)
    print(f"    mean absolute optimism {np.mean(opt):+.4f}, "
          f"largest {np.max(opt):+.4f}")

    print(f"\n    {'contrast':<40}{'row split':>11}{'held out':>10}"
          f"{'diff':>8}{'spread':>8}")
    diffs = []
    for k, mc in ch["mean_contrasts"].items():
        d_row = next(c["delta_accuracy"] for c in out["contrasts"]
                     if c["contrast"] == k)
        d_ch = mc["delta_accuracy_mean"]
        mc["delta_accuracy_row_split"] = round(d_row, 6)
        mc["difference_from_row_split"] = round(d_ch - d_row, 6)
        mc["sign_preserved"] = bool(np.sign(d_row) == np.sign(d_ch))
        diffs.append(abs(d_ch - d_row))
        print(f"    {mc['meaning']:<40}{d_row:+11.4f}{d_ch:+10.4f}"
              f"{d_ch - d_row:+8.4f}{mc['fold_spread']:8.4f}")

    signs_ok = all(m["sign_preserved"] for m in ch["mean_contrasts"].values())
    ch["all_signs_preserved"] = signs_ok
    ch["max_abs_difference"] = round(float(np.max(diffs)), 6)
    ch["mean_abs_difference"] = round(float(np.mean(diffs)), 6)

    # The optimism is read against a scale rather than declared. The mean gap is
    # a small fraction of the design-based half-width on the rungs it is being
    # compared with, and one rung's optimism is NEGATIVE - L4 scores better on
    # unseen clusters - which a flat "the row-wise split IS optimistic" cannot be
    # describing. How many rungs clear their own interval is counted below and
    # written into the verdict, so the sentence is falsifiable from the numbers
    # beside it.
    #
    # No count in this comment. It used to say "only one rung's gap clears its own
    # interval", which was true when it was typed and is not true now; the number
    # it was describing is `exceeds`, three lines down.
    #
    # `> half[r] > 0` and not just `> half[r]`: L7_hgb_demo's interval half-width
    # is exactly zero - hemoglobin plus demographics reproduces the label on every
    # replicate - so any positive gap at all would "exceed" it and the rung would
    # be reported as the one that fails, on a comparison that has no scale.
    half = {r: (out["rungs"][r]["binary"]["test_accuracy_ci95"][1]
                - out["rungs"][r]["binary"]["test_accuracy_ci95"][0]) / 2.0
            for r in ch["mean_accuracy_by_rung"]}
    per_rung = {r: round(out["rungs"][r]["binary"]["test_accuracy"] - a, 6)
                for r, a in ch["mean_accuracy_by_rung"].items()}
    exceeds = [r for r, g in per_rung.items() if g > half[r] > 0]
    negative = [r for r, g in per_rung.items() if g < 0]
    ch["optimism_by_rung"] = per_rung
    ch["optimism_vs_own_ci_half_width"] = {
        r: (round(per_rung[r] / half[r], 3) if half[r] > 0 else None)
        for r in per_rung}
    ch["rungs_where_optimism_exceeds_own_ci"] = exceeds
    ch["rungs_with_negative_optimism"] = negative
    ch["verdict"] = (
        f"Absolute accuracy falls by {ch['absolute_optimism_mean']:+.4f} on "
        f"average ({ch['absolute_optimism_max']:+.4f} at worst) when the test "
        f"clusters are genuinely unseen. That is small against the design-based "
        f"intervals on the same rungs: the gap exceeds a rung's own 95% "
        f"half-width for "
        + (f"{len(exceeds)} of {len(per_rung)} rungs ({', '.join(exceeds)})"
           if exceeds else f"none of the {len(per_rung)} rungs")
        + (f", and it is negative for {', '.join(negative)}, which scores "
           f"HIGHER on unseen clusters" if negative else "")
        + f". So the direction is toward optimism on average and it is not a "
          f"uniform property of the rungs; the earlier version of this string "
          f"asserted flatly that the row-wise split IS optimistic, from a mean "
          f"of {ch['absolute_optimism_mean']:+.4f}. A fold also trains on fewer "
          f"rows than dp.split3 does, so even the mean is an upper bound on "
          f"cluster optimism rather than a measurement of it. The six paired "
          f"contrasts move by {ch['mean_abs_difference']:.4f} on average and "
          f"{ch['max_abs_difference']:.4f} at worst, and "
        + ("every sign is preserved" if signs_ok
           else "AT LEAST ONE SIGN FLIPS, which would invalidate the "
                "cancellation argument")
        + ". That is the measured version of the claim in dp.split3's "
          "docstring: cluster optimism is common to both arms of a contrast and "
          "largely cancels, while it does not cancel in an absolute accuracy.")
    print(f"\n    {ch['verdict']}")

    # ------------------------------------------------------ tracking claim --
    out["tracking"] = tracking(out["rungs"])
    tk = out["tracking"]
    print("\n\nDOES ACCURACY TRACK RECOVERABILITY?")
    for key, lab in [("fixed_scope_no_demographics", "within a fixed feature scope"),
                     ("all_rungs", "across the whole ladder")]:
        t = tk[key]
        if t:
            print(f"  {lab:<32} Spearman rho {t['spearman_rho']:+.3f}  "
                  f"p {t['p_value']:.4f}  (n = {t['n']} rungs)")
    print(f"  {tk['claim']}")
    l0 = out["rungs"]["L0_hgb_direct"]["binary"]["test_accuracy"]
    l6 = out["rungs"]["L6_identity_free_demo"]["binary"]["test_accuracy"]
    print(f"  the counterexample: L0 has hemoglobin verbatim and scores "
          f"{l0:.4f}, below L6 at {l6:.4f}")

    # ------------------------------------------------------- decomposition --
    print("\n\nDECOMPOSING THE SUBMITTED 99%")
    out["decomposition"] = decompose(df, tr, te, strata, psu)
    dc = out["decomposition"]
    print(f"  {'features':<20}{'label':<24}{'acc':>8}{'NIR':>8}{'lift':>8}"
          f"{'PR-AUC':>9}{'no-skill':>10}")
    for r in dc["cells"]:
        print(f"  {r['features']:<20}{r['label']:<24}{r['accuracy']:8.4f}"
              f"{r['no_information_rate']:8.4f}{r['lift_points']:7.2f}p"
              f"{r['pr_auc']:9.4f}{r['pr_auc_no_skill']:10.4f}")
    e = dc["effects_on_lift_scale"]
    print("\n  on the lift scale (accuracy minus that cell's own NIR):")
    print(f"    identity removal, old sex-blind label   {e['identity_removal_under_old_label']:+6.2f} points")
    print(f"    identity removal, WHO label             {e['identity_removal_under_who_label']:+6.2f} points")
    print(f"    label correction, identity present      {e['label_correction_with_identity']:+6.2f} points")
    print(f"    label correction, identity gone         {e['label_correction_without_identity']:+6.2f} points")
    print(f"    interaction                             {e['interaction']:+6.2f} points")
    print(f"    leaky+old vs clean+WHO, total           {e['total_leaky_old_vs_clean_who']:+6.2f} points")
    print(f"  {dc['note']}")
    for k, t in dc["tests"].items():
        print(f"    {k:<38} delta {t['delta_accuracy']:+.4f} "
              f"[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}]  p {t['p_value']:.3g}  "
              f"Holm p {t['p_holm']:.3g}")
    print(f"    {dc['significance']['note']}")

    # -------------------------------------------------- out-of-cycle check --
    print("\n\nOUT-OF-CYCLE VALIDATION  (models fitted on 2017-2020 only)")
    out["out_of_cycle"] = out_of_cycle(models)
    key = ["L1_paper_leaky", "L4_identity_free", "L6_identity_free_demo"]
    print(f"  {'cycle':<12}{'n':>7}{'prev':>7}{'NIR':>8}"
          + "".join(f"{k.split('_')[0]:>10}" for k in key) + "   lift over NIR")
    for cycle, res in out["out_of_cycle"].items():
        nir = res["rungs"][key[0]]["no_information_rate"]
        accs = "".join(f"{res['rungs'][k]['accuracy']:>10.4f}" for k in key)
        lifts = " ".join(f"{res['rungs'][k]['lift_points']:+.2f}" for k in key)
        print(f"  {cycle:<12}{res['n']:>7,}{res['prevalence']*100:>6.2f}%"
              f"{nir:>8.4f}{accs}   {lifts}")
    tenir = maj_bin
    accs = "".join(f"{out['rungs'][k]['binary']['test_accuracy']:>10.4f}" for k in key)
    lifts = " ".join(f"{(out['rungs'][k]['binary']['test_accuracy']-tenir)*100:+.2f}"
                     for k in key)
    print(f"  {'2017-2020':<12}{len(te):>7,}{yte_bin.mean()*100:>6.2f}%"
          f"{tenir:>8.4f}{accs}   {lifts}   <- held-out rows, same cycle")
    # The reproducibility claim, measured. It was two hardcoded print lines
    # asserting "the ordering L1 > L4 and L6 > L4 reproduces in every cycle",
    # which happens to be true and hides that one margin is a fifth of the
    # others: 2015-2016's L1 - L4 gap is +0.0023 against +0.0153 and +0.0177 in
    # the neighbouring cycles. A claim of reproducibility that cannot show its
    # smallest margin is not showing the thing that would falsify it.
    ordering = {}
    for pair in [("L1_paper_leaky", "L4_identity_free"),
                 ("L6_identity_free_demo", "L4_identity_free")]:
        gaps = {c: round(res["rungs"][pair[0]]["accuracy"]
                         - res["rungs"][pair[1]]["accuracy"], 6)
                for c, res in out["out_of_cycle"].items()}
        gaps[PRIMARY] = round(
            out["rungs"][pair[0]]["binary"]["test_accuracy"]
            - out["rungs"][pair[1]]["binary"]["test_accuracy"], 6)
        ordering[f"{pair[0]} > {pair[1]}"] = {
            "gap_by_cycle": gaps,
            "holds_in_all_cycles": bool(all(v > 0 for v in gaps.values())),
            "cycles_where_it_holds": sorted(c for c, v in gaps.items() if v > 0),
            "smallest_margin": min(gaps.values()),
            "smallest_margin_cycle": min(gaps, key=gaps.get),
        }
    out["out_of_cycle_ordering"] = ordering
    print()
    for name, rec in ordering.items():
        verb = ("holds in all "
                f"{len(rec['gap_by_cycle'])} cycles" if rec["holds_in_all_cycles"]
                else f"holds in {len(rec['cycles_where_it_holds'])} of "
                     f"{len(rec['gap_by_cycle'])} cycles")
        print(f"  {name:<44} {verb}; smallest margin "
              f"{rec['smallest_margin']:+.4f} ({rec['smallest_margin_cycle']})")
    if all(r["holds_in_all_cycles"] for r in ordering.values()):
        print("  the ladder's ordering is a property of the CBC and not of one "
              "sample -- but read")
        print("  the smallest margins above before calling any single cycle "
              "confirmatory.")

    l6r, l7r = out["rungs"]["L6_identity_free_demo"], out["rungs"]["L7_hgb_demo"]
    print("\nTHE COMPLETE LEAK")
    print(f"  L7 hemoglobin + age/sex/pregnancy  accuracy "
          f"{l7r['binary']['test_accuracy']:.4f}  PR-AUC {l7r['binary']['pr_auc']:.4f}")
    print("  Nothing is predicted: the WHO label is a deterministic function of")
    print("  these four inputs. It is the upper bound the ladder is measured")
    print("  against, and it is the feature design of the prior work whose target")
    print("  is itself a threshold rule over the same columns.")

    print("\nTHE DEFENSIBLE MODEL")
    print(f"  L6 identity-free + demographics    accuracy "
          f"{l6r['binary']['test_accuracy']:.4f}  PR-AUC {l6r['binary']['pr_auc']:.4f}")
    print(f"  no-information rate                         {maj_bin:.4f}")
    print(f"  lift over the floor                         "
          f"{(l6r['binary']['test_accuracy']-maj_bin)*100:.2f} points")
    print("  and it still holds up out of cycle: "
          + ", ".join(f"{c} {r['rungs']['L6_identity_free_demo']['lift_points']:+.2f}p"
                      for c, r in out["out_of_cycle"].items()))

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
