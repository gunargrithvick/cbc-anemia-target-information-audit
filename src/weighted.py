"""
Step 5 - survey-design-aware prevalence and model metrics.

NHANES is not a simple random sample. It oversamples some groups on purpose,
and it selects people in clusters within strata. Three consequences that a
simple row-wise analysis can miss are:

  1. PREVALENCE. Unweighted 9.60% is a property of the sample, not of the
     United States. The population figure is lower, because the groups NHANES
     oversamples are the ones with more anemia.
  2. STANDARD ERRORS. The naive binomial SE is too small, so every
     "significant" difference computed from it is overconfident. The design
     effect on weighted prevalence is about 4.0, but it is not a clustering
     effect: it factors into 2.42 from unequal weighting (the Kish factor
     1 + CV^2) times 1.65 from clustering. Both are reported, because
     "clustering costs a factor of 4" is the wrong sentence and it is the one
     everybody writes. Nor does it transfer to the paired accuracy contrasts
     in ablation.py, where the measured Rao-Wu penalty is 0.90x.
  3. MODEL PERFORMANCE. Recall measured on the unweighted sample is not the
     recall a screening tool would show in the population it is deployed on.

This script computes all three with a proper Taylor-series (linearised)
variance estimator over SDMVSTRA / SDMVPSU, which is what SUDAAN, Stata's
svy: and R's survey package do.

Two things here are answers to the obvious objections.

  A weighted point estimate without a design-based interval is not an
  improvement on an unweighted one, so every model metric in section 5 carries
  the same linearised SE and the same 25 degrees of freedom as the prevalence
  figures. Accuracy, sensitivity, specificity and precision are all weighted
  proportions over a domain, which is exactly what svy_prop estimates; PR-AUC is
  not, so it gets a Rao-Wu rescaled bootstrap instead. The test that matters is
  whether the unweighted number a conventional paper would report falls inside
  the population interval, and that is reported per metric rather than asserted.

  Weights can enter twice, at scoring and at fitting, and they are usually
  confused. Scoring with weights answers "what would this model do in the
  population"; fitting with weights answers "what model does the population
  ask for". Section 6 does both and reports the difference, because claiming
  the survey design matters while never letting it touch the estimator would
  be the same omission this script criticises.

Outputs
  results/weighted.json
  stdout   the prevalence, design and design-based model tables for the paper

Run:  python src/weighted.py
"""

import json
import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold

import dataprep as dp
from paths import RESULTS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = dp.SEED
N_FOLDS = dp.N_FOLDS       # one place, dataprep; see the note there
N_REP = dp.N_REP           # one place, dataprep; see the note there. Used here
                           # for the metrics that are not proportions and so
                           # have no linearised form.
WEIGHT = dp.MECWT          # MEC exam weight: the CBC is a MEC lab measure.
                           # dataprep renames the cycle-specific source column
                           # (WTMECPRP here, WTMEC2YR earlier) to one name so
                           # nothing downstream can silently use the wrong one.
STRATUM = dp.STRATUM
PSU = dp.PSU
RUNG = "L6_identity_free_demo"
OUT_PATH = RESULTS / "weighted.json"


def weight_cv(w):
    """
    Coefficient of variation of a weight vector, on the Kish convention (ddof=0).

    One function because this was computed twice with two conventions and both
    numbers were published. svy_prop derived it from the Kish factor as
    sqrt(n/n_eff - 1), and the design block computed w.std(ddof=1)/w.mean()
    directly, so weighted.json carried 1.1904091 in one place and 1.1903602 in
    the other for the same weights on the same rows. The gap is exactly
    sqrt(n/(n-1)) - harmless in size, and precisely the kind of thing a reviewer
    finds and a reader cannot resolve.

    ddof=0 is the convention that has to win, because the identity the paper
    quotes is the Kish factor n/n_eff = 1 + CV^2(w), and that identity holds for
    the population-form variance only. Using ddof=1 here would make the printed
    "Kish factor 1 + CV^2" line not reproduce the DEFF component printed beside
    it.
    """
    w = np.asarray(w, dtype=float)
    m = w.mean() if len(w) else 0.0
    return float(w.std(ddof=0) / m) if m else None


def cluster_optimism_from_ablation(half_width):
    """
    The shared-cluster optimism disclosure, read from ablation.json, not re-typed.

    This block used to carry three literals - 0.002, 0.006 and the string "roughly
    0.4x to 1.1x" - all three of which are ablation.cluster_holdout's numbers
    copied across a file boundary by hand. By the time anyone checked, the ratio
    was wrong at the bottom end: the mean optimism is 0.32x of this block's own
    interval half-width, not 0.4x. That is the whole failure mode of a re-typed
    number, and it was sitting in the honesty disclosure.

    Both the magnitude and the ratio are now computed. The magnitude comes from
    ablation.json, which the pipeline writes before this script runs; the ratio is
    computed against the half-width of the interval printed a few lines below, so
    "smaller than the interval" is a claim this function can be held to rather
    than an assertion.

    An absent ablation.json is recorded as absent. Falling back to a remembered
    0.002 would publish a number with no run behind it under a key whose entire
    purpose is to say where the number came from.
    """
    p = RESULTS / "ablation.json"
    if not p.exists():
        return {"available": False,
                "reason": f"{p.name} is not in results/, so the magnitude of "
                          f"this optimism has not been measured on this run; "
                          f"run ablation.py and re-run this script"}
    ch = json.loads(p.read_text(encoding="utf-8")).get("cluster_holdout") or {}
    mean = ch.get("absolute_optimism_mean")
    worst = ch.get("absolute_optimism_max")
    if mean is None or worst is None:
        return {"available": False,
                "reason": "ablation.json has no cluster_holdout block with "
                          "absolute_optimism_mean and absolute_optimism_max"}
    ratio = None
    if half_width and np.isfinite(half_width) and half_width > 0:
        ratio = (f"{mean / half_width:.2f}x to {worst / half_width:.2f}x of the "
                 f"design-based 95% half-width for this estimate "
                 f"({half_width:.6f})")
    return {"available": True,
            "accuracy_points_mean": float(mean),
            "accuracy_points_worst_case": float(worst),
            "relative_to_ci_half_width": ratio,
            "exceeds_ci_half_width": (None if ratio is None
                                      else bool(worst > half_width))}


def svy_prop(y, w, strata, psu, domain=None, alpha=0.05):
    """
    Design-based proportion with a linearised (Taylor-series) standard error.

    The estimate is a ratio, p = sum(w*d*y) / sum(w*d), so its variance is the
    variance of the linearised residual u = w*d*(y - p) accumulated over PSUs
    within strata:

        var(p) = (1 / X^2) * SUM_h [ n_h/(n_h-1) * SUM_i (u_hi - ubar_h)^2 ]

    where h indexes strata, i indexes PSUs inside a stratum, X = sum(w*d) and
    n_h is the number of PSUs in stratum h. Degrees of freedom = (#PSUs) -
    (#strata), the standard NHANES convention.

    domain lets you estimate for a subgroup WITHOUT subsetting the data, which
    is required: dropping rows would discard strata and understate variance.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    d = np.ones_like(w) if domain is None else np.asarray(domain, dtype=float)

    X = float((w * d).sum())
    if X <= 0:
        # An empty domain is a defined situation with an undefined estimate, and
        # it used to return a bare None that main() then subscripted. It is
        # reachable: precision_ppv's domain is "everyone the model flagged", so
        # a model that flags nobody lands here and the run died on
        # r["ci95_logit"] rather than saying what happened.
        return {"undefined": True,
                "reason": "the domain has zero weighted total: no sampled "
                          "person satisfies it, so this proportion has no "
                          "denominator",
                "n_obs": 0, "prop_weighted": None, "prop_unweighted": None,
                "se": None, "ci95_logit": None, "ci95_wald": None,
                "design_effect": None}
    p = float((w * d * y).sum() / X)

    u = w * d * (y - p)                 # linearised residual, 0 outside domain
    var = 0.0
    n_psu_total = 0
    n_strata = 0
    singletons = []
    for h in np.unique(strata):
        hm = strata == h
        psus = np.unique(psu[hm])
        n_h = len(psus)
        n_strata += 1
        n_psu_total += n_h
        if n_h < 2:
            # A lone PSU has no within-stratum spread to measure, so it adds
            # zero and the SE comes out too SMALL - the failure is in the
            # anti-conservative direction, which is why it must not pass
            # unremarked. It is recorded in singleton_strata and printed. This
            # design has none (24 strata, every one with 2 or 3 PSUs, and the
            # domain trick keeps all PSUs regardless of subgroup), but a future
            # cycle could, and the standard remedy is collapsing strata, not
            # ignoring them.
            singletons.append(int(h))
            continue
        tot = np.array([u[hm & (psu == q)].sum() for q in psus])
        var += n_h / (n_h - 1.0) * ((tot - tot.mean()) ** 2).sum()
    var /= X ** 2
    se = float(np.sqrt(max(var, 0.0)))
    df = max(n_psu_total - n_strata, 1)
    t = float(stats.t.ppf(1 - alpha / 2, df))

    # Wald CI can leave [0,1] for rare outcomes; the logit-transformed interval
    # cannot, which is why NCHS recommends it for low-prevalence estimates.
    lo_w, hi_w = p - t * se, p + t * se
    if 0 < p < 1 and se > 0:
        g = se / (p * (1 - p))          # SE on the logit scale, delta method
        l = np.log(p / (1 - p))
        lo_l = 1 / (1 + np.exp(-(l - t * g)))
        hi_l = 1 / (1 + np.exp(-(l + t * g)))
    else:
        lo_l, hi_l = lo_w, hi_w

    n_obs = int(d.sum())
    in_dom = d > 0
    wd = w[in_dom]
    kish = float(wd.sum() ** 2 / (wd ** 2).sum()) if len(wd) else 0.0
    se_srs = float(np.sqrt(p * (1 - p) / n_obs)) if n_obs else 0.0
    deff = float((se / se_srs) ** 2) if se_srs > 0 else None

    # A second naive SE, and the reason there are two.
    #
    # se_srs_naive is the binomial SE evaluated at the WEIGHTED proportion, which
    # is the right denominator for a design effect: DEFF has to compare two
    # variances of the SAME estimand, or it is not a design effect. But it is NOT
    # the interval a conventional paper actually prints. That paper never computes
    # a weighted proportion at all - it reports p-hat unweighted and puts a
    # binomial interval around it, so its SE is evaluated at p-hat.
    #
    # The two differ here because weighting moves prevalence: 9.60% unweighted
    # against 6.63% weighted, and sqrt(p(1-p)) is not flat between them. So
    # sqrt(DEFF) = 2.00 answers "how much does the design inflate the variance of
    # the population proportion", while 1.69 answers "how much wider is the honest
    # interval than the one in the paper next door". The stdout used to quote the
    # first number for the second question, which overstates the correction by
    # about 18%. Both are recorded, each against its own question.
    p_un = float(y[in_dom].mean()) if n_obs else None
    se_srs_un = (float(np.sqrt(p_un * (1 - p_un) / n_obs))
                 if n_obs and p_un is not None else None)
    ratio_vs_paper = (float(se / se_srs_un) if se_srs_un else None)

    # The design effect splits, and it was being described as though it did not.
    #
    # DEFF near 4 was reported as what "the clustering costs". It is not. The
    # unequal-weighting part is the Kish factor n/n_eff = 1 + CV^2(w), and here
    # that alone is about 2.42; the residual DEFF/2.42, about 1.65, is what the
    # clustering costs. So clustering widens an interval by sqrt(1.65) = 1.29x,
    # not the 2.0x the total implies, and the dominant term is that NHANES
    # oversamples on purpose. Both components are reported so the sentence in
    # the paper can be the true one.
    deff_weights = float(n_obs / kish) if kish > 0 else None
    deff_cluster = (float(deff / deff_weights)
                    if deff and deff_weights else None)
    # weight_cv() rather than sqrt(deff_weights - 1): algebraically the same
    # quantity, but routed through the one definition so it cannot disagree with
    # the design block's copy. It is also defined when kish is 0.
    cv_w = weight_cv(wd)

    # The note used to end with a flat "attributing the whole design effect to
    # clustering overstates it by more than a factor of two". That is true of the
    # whole-cohort prevalence row, which is the row it was written from, and false
    # of 5 of the 24 estimates this function produces: pregnant women (1.670),
    # children 1-4 (1.500), 5-11 (1.816), 12-14 (1.882) and age 70+ (1.774) all
    # come in under 2. A boilerplate sentence that is wrong on a fifth of the rows
    # it is attached to is worse than no sentence, so the factor is now computed
    # from this row's own numbers.
    if deff and deff_weights:
        over = f"{deff / deff_cluster:.2f}" if deff_cluster else "?"
        note = ("design_effect = deff_unequal_weighting x "
                "deff_clustering_residual. The first is the Kish factor "
                "1 + CV^2(w) and comes from deliberate oversampling; the second "
                f"is what the clustering costs. For THIS estimate, attributing "
                f"the whole design effect to clustering overstates it by {over}x "
                f"- read deff_unequal_weighting, not a remembered number from "
                f"another row.")
    else:
        note = ("the design effect is undefined for this domain, so it does not "
                "decompose")

    # deff < 1 happens, and it is not a bug: the linearised design SE can come out
    # below the binomial SRS SE when a domain's outcome is close to homogeneous
    # inside PSUs, or when a small domain's weighted total is dominated by a few
    # PSUs that happen to agree. It IS a thing a reader should be told about
    # rather than left to notice, because "the design effect is 0.90" reads as an
    # arithmetic error. 2 of the 24 estimates here are below 1: pregnant women at
    # 0.901 and severity.Severe at 0.365. The illustrative figure used to be a
    # made-up 0.83, which sends a reader looking through the JSON for a number
    # that is not in it.
    deff_below_1 = bool(deff is not None and deff < 1.0)

    # n_obs / deff exceeds n_obs whenever deff < 1, which is how
    # severity.Severe reported an effective sample of 33,343 from 12,156 rows -
    # a number that cannot mean what its name says. Capped, with the raw value
    # kept beside it so nothing is hidden.
    n_eff_design_raw = float(n_obs / deff) if deff else None
    n_eff_design = (min(n_eff_design_raw, float(n_obs))
                    if n_eff_design_raw is not None else None)

    return {
        "n_obs": n_obs,
        "prop_weighted": p,
        "prop_unweighted": float(y[in_dom].mean()) if n_obs else None,
        "se": se,
        "df": int(df),
        "ci95_wald": [float(lo_w), float(hi_w)],
        "ci95_logit": [float(lo_l), float(hi_l)],
        "se_srs_naive": se_srs,
        "se_srs_naive_at_unweighted_p": se_srs_un,
        "se_ratio_vs_conventional_binomial": ratio_vs_paper,
        "se_ratio_note":
            "design_effect and its square root are the inflation of the "
            "POPULATION proportion's variance, both SEs taken at the weighted p. "
            "se_ratio_vs_conventional_binomial is the different, smaller question "
            "a reader actually asks: how much wider the design interval is than "
            "the binomial interval a conventional paper puts around its "
            "UNWEIGHTED p-hat. Quoting sqrt(design_effect) for that overstates "
            "it, because weighting moves the proportion and sqrt(p(1-p)) is not "
            "flat between the two values.",
        "design_effect": deff,
        "deff_unequal_weighting": deff_weights,
        "deff_clustering_residual": deff_cluster,
        "deff_below_1": deff_below_1,
        "deff_below_1_note":
            ("the design SE came out BELOW the naive binomial SE for this "
             "domain, so the design 'penalty' is a gain here. Small domains "
             "whose outcome varies little within PSUs do this; it is not an "
             "arithmetic error, and it is why n_effective_design is capped at "
             "n_obs." if deff_below_1 else None),
        "weight_cv": cv_w,
        "deff_decomposition_note": note,
        "n_effective_design": n_eff_design,
        "n_effective_design_uncapped": n_eff_design_raw,
        "n_effective_kish": kish,
        "population_total": float((w * d).sum()),
        "population_with_outcome": float((w * d * y).sum()),
        "population_note":
            "complete-case: these totals are what the 12,156 analysed rows "
            "represent, not the whole NHANES frame. See "
            "complete_case_population_excluded in the design block.",
        "singleton_strata": singletons,
        "singleton_strata_note":
            (f"{len(singletons)} stratum/strata contributed no variance because "
             f"they hold a single PSU, so this SE is a LOWER bound. Collapse "
             f"them before reporting." if singletons else
             "None: every stratum has at least 2 PSUs, so no variance "
             "contribution was dropped."),
    }


def domains(df):
    """The subgroups the paper should report, as boolean masks."""
    age, sex = df[dp.AGE].to_numpy(), df[dp.SEX].to_numpy()
    preg = df[dp.PREG].to_numpy()
    return {
        "all": np.ones(len(df), bool),
        "men 15+": (sex == 1) & (age >= 15),
        "women 15+ non-pregnant": (sex == 2) & (age >= 15) & (preg == 0),
        "pregnant": (preg == 1),
        "children 1-4": age <= 4,
        "children 5-11": (age >= 5) & (age <= 11),
        "children 12-14": (age >= 12) & (age <= 14),
        "age 15-49": (age >= 15) & (age <= 49),
        "age 50-69": (age >= 50) & (age <= 69),
        "age 70+": age >= 70,
    }


def oof_predictions(X, y, fit_weight=None):
    """
    Out-of-fold predictions and probabilities over the whole cohort.

    Written out by hand rather than with cross_val_predict because section 6
    needs to pass sample_weight to fit, and cross_val_predict cannot route a
    fit parameter without sklearn's global metadata routing switched on. Same
    folds, same seed and same estimator either way, so the two runs differ only
    in whether the forest was told about the survey weights.

    class_weight="balanced" stays on in both runs. With fit_weight supplied the
    two multiply: each row's contribution is its survey weight times its class
    weight. That is the intended combination -- the survey weight says how many
    people the row stands for, the class weight says how much a rare outcome
    counts -- but it does mean the weighted fit is not simply "the same model on
    a reweighted sample".
    """
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y), dtype=float)
    cvs = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cvs.split(X, y):
        clf = RandomForestClassifier(n_estimators=dp.N_TREES,
                                     class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        kw = {} if fit_weight is None else {"sample_weight": fit_weight[tr]}
        clf.fit(X[tr], y[tr], **kw)
        pred[te] = clf.predict(X[te])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return pred, proba


def svy_metrics(y, pred, w, strata, psu):
    """
    The confusion-matrix metrics as design-based weighted proportions.

    Every one of them is a proportion over a domain, so all four come out of the
    same linearised ratio estimator as prevalence, on the same degrees of
    freedom -- no second machinery and no bootstrap required:

        accuracy      P(pred = y)             domain: everyone
        sensitivity   P(pred = 1 | y = 1)     domain: the anemic
        specificity   P(pred = 0 | y = 0)     domain: the non-anemic
        precision     P(y = 1 | pred = 1)     domain: those flagged

    Domain estimation rather than subsetting matters here: taking only the
    anemic rows would delete whole PSUs from some strata and shrink the
    variance, which is the same mistake as ignoring the design in the first
    place.
    """
    spec = [("accuracy", (pred == y).astype(int), None),
            ("sensitivity_recall", pred, y == 1),
            ("specificity", 1 - pred, y == 0),
            ("precision_ppv", y, pred == 1)]
    out = {}
    for name, outcome, dom in spec:
        out[name] = svy_prop(outcome, w, strata, psu,
                             domain=None if dom is None
                             else dom.astype(float))
    return out


def _raowu(stat, theta, w, strata, psu, n, alpha):
    """
    The Rao-Wu replicate loop, once.

    svy_pr_auc and paired_design_test each carried their own copy of this: fresh
    RNG at SEED, n draws of dp.rao_wu_weights(base=w), drop the non-finite ones,
    centre the spread on the OBSERVED statistic rather than on the replicate
    mean, and take a t interval on the design df. Two copies of a variance
    estimator is two chances to fix a bug in one of them, and the difference
    would show up as two intervals in one file that no longer used the same
    scheme while still both being labelled "Rao-Wu rescaled bootstrap".

    Centring on theta, not on vals.mean(), is deliberate and is why this is
    sqrt(mean(square)) rather than a variance: the replicate estimator's target
    is the mean squared deviation of the replicates from the full-sample
    estimate.

    A fresh default_rng(SEED) per call is also deliberate. It gives every
    statistic in this file the same replicate sequence, which is common random
    numbers: the paired contrasts stay paired, and PR-AUC is measured against
    the same draws as everything else.
    """
    rng = np.random.default_rng(SEED)
    vals = np.array([stat(dp.rao_wu_weights(strata, psu, rng, base=w))
                     for _ in range(n)], dtype=float)
    vals = vals[np.isfinite(vals)]
    se = float(np.sqrt(np.mean((vals - theta) ** 2)))
    df = int(dp.design_df(strata, psu))
    t = float(stats.t.ppf(1 - alpha / 2, df))
    return se, df, [theta - t * se, theta + t * se], int(len(vals))


def svy_pr_auc(y, proba, w, strata, psu, n=N_REP, alpha=0.05):
    """
    Weighted PR-AUC with a Rao-Wu rescaled bootstrap interval.

    Average precision is not a ratio of weighted totals, so the linearised
    estimator does not apply to it and pretending otherwise would be the kind of
    shortcut this project is about. The Rao-Wu scheme does apply: draw
    m_h = n_h - 1 PSUs with replacement per stratum and rescale, which at two
    PSUs per stratum is BRR half-sample logic and has the right variance
    expectation, unlike the naive n_h-of-n_h cluster bootstrap.
    """
    theta = float(average_precision_score(y, proba, sample_weight=w))
    se, df, ci, reps = _raowu(
        lambda ww: average_precision_score(y, proba, sample_weight=ww),
        theta, w, strata, psu, n, alpha)
    return {"prop_weighted": theta, "se": se, "df": df,
            "ci95": ci,
            "prop_unweighted": float(average_precision_score(y, proba)),
            "replicates": reps,
            "variance_estimator": "Rao-Wu rescaled bootstrap"}


def paired_design_test(stat, strata, psu, w, n=N_REP, alpha=0.05):
    """
    A paired difference of two weighted statistics on the same rows.

    stat(weights) must return the difference itself, so the pairing is exact:
    replicate b reweights both models identically and the two share every row.

    The degenerate cases come from dataprep.t_and_p rather than from a local
    `obs / se if se > 0 else np.inf`, which is what this line used to be. That
    version reported p = 0 for two models with identical predictions, where
    ablation.py and benchmark.py both reported p = 1 for the same situation, and
    it could put float("inf") into a JSON written with allow_nan=False.
    """
    obs = float(stat(w))
    se, df, ci, _ = _raowu(stat, obs, w, strata, psu, n, alpha)
    tp = dp.t_and_p(obs, se, df)
    return {"delta": obs, "se": se, "df": df, "ci95": ci, **tp}


def main():
    df, att = dp.load_cohort()
    w = df[WEIGHT].to_numpy()
    st = df[STRATUM].to_numpy()
    ps = df[PSU].to_numpy()
    y = df["Anemia"].to_numpy()

    out = {"cohort": att, "weight_variable": WEIGHT,
           "weight_source_column": att["weight_column"],
           # benchmark.json records seed / n_folds / n_replicates at top level and
           # ablation.json records cluster_holdout.seed and
           # variance_scales.replicates. This file recorded none of them, so its
           # numbers - including the headline population accuracy - could not be
           # reproduced from the JSON alone. Every stochastic input this script
           # has, named.
           "seed": SEED,
           "n_folds": N_FOLDS,
           "n_replicates": N_REP,
           "provenance_note":
               "seed is dp.SEED, used for the StratifiedKFold shuffle, both "
               "forests and the Rao-Wu replicate generator. n_replicates applies "
               "to PR-AUC and to the three fit-weight contrasts; every other "
               "metric here is a linearised ratio estimate with no resampling."}

    # ------------------------------------------------------------- design ----
    per_stratum = {int(h): int(len(np.unique(ps[st == h]))) for h in np.unique(st)}
    # population_represented keeps its name because results/weighted.json is read
    # by figures.py and by the results contract under that key, but it is a
    # COMPLETE-CASE total and used to be printed as though it were the frame. The
    # 1,616 excluded rows carry weight too; dp.attrition_profile now measures how
    # much, and it is carried here so the two numbers sit together instead of one
    # of them being a footnote in a different file.
    wl = att.get("attrition_profile", {}).get("weighted_loss", {})
    out["design"] = {
        "n_strata": len(per_stratum),
        "psus_per_stratum": per_stratum,
        "total_psus": int(sum(per_stratum.values())),
        "df": int(sum(per_stratum.values()) - len(per_stratum)),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "weight_cv": weight_cv(w),
        "population_represented": float(w.sum()),
        "complete_case_population_excluded": wl.get("population_dropped"),
        "complete_case_population_excluded_pct": wl.get("population_dropped_pct"),
        "population_represented_note":
            ("population_represented is the total for the "
             f"{att['n_analysis']:,} analysed rows only. The "
             f"{att['excluded_total']:,} rows dropped for an incomplete CBC "
             f"represent a further "
             f"{(wl.get('population_dropped') or 0)/1e6:.1f} million "
             f"({wl.get('population_dropped_pct')}% of the merged frame), whose "
             f"anemia burden is unmeasurable because they have no CBC. Every "
             f"population count in this file is therefore a lower bound on the "
             f"frame."),
    }
    print("SURVEY DESIGN")
    print(f"  weight variable                  {att['weight_column']} "
          f"(MEC exam weight, read as {WEIGHT})")
    print(f"  strata                           {len(per_stratum)}")
    print(f"  PSUs total                       {sum(per_stratum.values())}")
    print(f"  degrees of freedom               {out['design']['df']}")
    print(f"  weight range                     {w.min():,.0f} to {w.max():,.0f}")
    print(f"  weight CV                        {out['design']['weight_cv']:.3f}")
    print(f"  population represented           {w.sum()/1e6:.1f} million "
          f"(complete case)")
    if wl.get("population_dropped"):
        print(f"  population NOT represented       "
              f"{wl['population_dropped']/1e6:.1f} million "
              f"({wl['population_dropped_pct']}% of the merged frame), from the "
              f"{att['excluded_total']:,} rows")
        print("  Those rows have no CBC, so their anemia burden cannot be "
              "measured, only bounded.")
        print("  Every population total below is a complete-case total, i.e. a "
              "lower bound.\n")
    else:
        print()
    # -------------------------------------------------- overall prevalence ---
    print("ANEMIA PREVALENCE, unweighted sample vs US population")
    out["prevalence"] = {}
    for lab, name in [("Anemia", "WHO age/sex/pregnancy"),
                      ("Anemia_hb12", "manuscript's single 12.0 g/dL cut")]:
        r = svy_prop(df[lab].to_numpy(), w, st, ps)
        out["prevalence"][lab] = {"label_definition": name, **r}
        print(f"  {name}")
        print(f"    unweighted                     {r['prop_unweighted']*100:6.2f}%")
        print(f"    weighted                       {r['prop_weighted']*100:6.2f}%"
              f"   SE {r['se']*100:.2f}")
        print(f"    95% CI (logit)                 "
              f"[{r['ci95_logit'][0]*100:.2f}%, {r['ci95_logit'][1]*100:.2f}%]")
        print(f"    naive binomial SE would be     {r['se_srs_naive']*100:.2f}"
              f"   -> design effect {r['design_effect']:.2f}")
        print(f"    effective sample size          "
              f"{r['n_effective_design']:,.0f} of {r['n_obs']:,}")
        print(f"    people represented as anemic   "
              f"{r['population_with_outcome']/1e6:.2f} million\n")

    a = out["prevalence"]["Anemia"]
    print(f"  The design effect of {a['design_effect']:.2f} means the naive "
          f"binomial SE understates the")
    print(f"  variance of the POPULATION proportion by "
          f"{np.sqrt(a['design_effect']):.2f}x - both SEs taken at the same "
          f"weighted p,")
    print("  which is the only comparison a design effect can be.")
    print("  Against the interval a conventional paper really prints - binomial, "
          "around its")
    print(f"  own unweighted {a['prop_unweighted']*100:.2f}% - the honest "
          f"interval is "
          f"{a['se_ratio_vs_conventional_binomial']:.2f}x wider, not "
          f"{np.sqrt(a['design_effect']):.2f}x.")
    print("  The two questions have two answers because weighting moves the "
          "proportion; this")
    print("  line used to give the first answer to the second question and "
          "overstate it.")
    print(f"  It does NOT all come from the clustering. It factors as "
          f"{a['deff_unequal_weighting']:.3f} from unequal")
    print(f"  weighting (the Kish factor 1 + CV^2, weight CV "
          f"{a['weight_cv']:.4f}) times "
          f"{a['deff_clustering_residual']:.3f} from")
    print(f"  clustering. Clustering alone widens an interval by "
          f"{np.sqrt(a['deff_clustering_residual']):.2f}x; the dominant term is")
    print("  that NHANES oversamples some groups on purpose. And none of it "
          "transfers to the")
    print("  paired accuracy contrasts: see variance_scales in ablation.json "
          "for the measured")
    print("  Rao-Wu penalty there. A design effect on a weighted PREVALENCE is "
          "not a design")
    print("  effect on an unweighted paired DIFFERENCE.\n")

    # ------------------------------------------------------------ subgroups --
    print("PREVALENCE BY SUBGROUP (domain estimation, full design retained)")
    hdr = (f"  {'group':<24}{'n':>7}{'unwtd':>8}{'wtd':>8}{'SE':>7}"
           f"{'95% CI':>18}{'pop (M)':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    out["subgroups"] = {}
    for name, mask in domains(df).items():
        r = svy_prop(y, w, st, ps, domain=mask)
        if r.get("undefined") or r["n_obs"] == 0:
            continue
        out["subgroups"][name] = r
        ci = (f"[{r['ci95_logit'][0]*100:.1f},"
              f"{r['ci95_logit'][1]*100:.1f}]")
        print(f"  {name:<24}{r['n_obs']:7,}{r['prop_unweighted']*100:7.2f}%"
              f"{r['prop_weighted']*100:7.2f}%{r['se']*100:7.2f}{ci:>18}"
              f"{r['population_total']/1e6:9.1f}")
    print()
    # ------------------------------------------------- severity prevalence ---
    print("WHO SEVERITY PREVALENCE (4-class), weighted")
    out["severity"] = {}
    sev = df["Severity4"].to_numpy()
    for i, nm in enumerate(dp.SEVERITY_NAMES_4):
        r = svy_prop((sev == i).astype(int), w, st, ps)
        out["severity"][nm] = r
        print(f"  {nm:<18}{r['n_obs'] and (sev == i).sum():>7,}"
              f"{r['prop_unweighted']*100:8.2f}%{r['prop_weighted']*100:8.2f}%"
              f"   SE {r['se']*100:.2f}   "
              f"{r['population_with_outcome']/1e6:6.2f}M")
    print()

    # ------------------------------------------- model performance, weighted -
    # Out-of-fold predictions over the WHOLE cohort, so the weighted metric can
    # use the full design instead of a 20% subsample of it.
    #
    # This is the one place in the project where the reported number does NOT
    # come from dp.split3, and the difference has to be stated rather than left
    # for a reader to find. Three files say some version of "every model here is
    # fitted on the train block alone" and "the test block is scored once"; those
    # sentences are about the ladder in ablation.py and the estimator comparison
    # in benchmark.py, and this block is the exception to both. It never calls
    # dp.split3 at all.
    #
    # Why keep it that way. The estimand is a POPULATION quantity - accuracy in
    # the United States, with a design-based interval on 25 degrees of freedom.
    # Scoring on the 2,432-row test block would throw away 80% of the weighted
    # sample and, worse, most of the design: the interval would be built from
    # whatever strata and PSUs happened to land in that fifth. Out-of-fold
    # prediction over all 12,156 rows keeps every stratum and every PSU and still
    # never scores a row with a model that saw it.
    #
    # What it costs, measured and not asserted. The folds are row-wise, so all 49
    # stratum-by-PSU cells appear in every training fold: a row's own cluster-
    # mates are in the model that predicts it. ablation.cluster_holdout measures
    # that same optimism directly by holding PSUs out instead of rows, and its
    # worst rung comes out slightly ABOVE this block's own interval half-width -
    # so it is not negligible, it does not change any conclusion here, and it is
    # disclosed in model_oof.shared_cluster_optimism below rather than in a
    # comment only.
    #
    # The magnitudes are not written here. They used to be, as "+0.002", "+0.006"
    # and a half-width of "0.0054", and two things had already drifted: the true
    # low end is 0.32x, not the 0.4x published beside it, and the disclosure's own
    # why_not_fixed string asserted the optimism was "smaller than the interval"
    # while the worst case is 1.08x of it. cluster_optimism_from_ablation reads all
    # of it out of ablation.json, divides by the interval this block actually
    # produced, and records the comparison as a boolean instead of a sentence.
    print("MODEL PERFORMANCE: sample vs population, with design-based intervals")
    print(f"  {N_FOLDS}-fold out-of-fold predictions over all {len(df):,} rows, "
          f"rung {RUNG}")
    print(f"  features: {', '.join(dp.LADDER[RUNG])}")
    print("  NOTE: this is the one block in the project that does not use "
          "dp.split3. The")
    print("  estimand is a population quantity, so it needs the whole design, "
          "not the 20%")
    print("  test block. The folds are row-wise, so a row's PSU-mates are in "
          "the model")
    print("  that predicts it; see shared_cluster_optimism in weighted.json.")
    X = df[dp.LADDER[RUNG]].to_numpy()
    pred, proba = oof_predictions(X, y)

    m = svy_metrics(y, pred, w, st, ps)
    m["pr_auc"] = svy_pr_auc(y, proba, w, st, ps)
    lo, hi = m["accuracy"]["ci95_wald"]
    out["model_oof"] = {"rung": RUNG, "features": dp.LADDER[RUNG],
                        "n_folds": N_FOLDS, "weights_used_in_fit": False,
                        "split": "5-fold out-of-fold over all rows; dp.split3 "
                                 "is deliberately not used here",
                        "shared_cluster_optimism": {
                            "mechanism": "row-wise folds put all 49 "
                                         "stratum-by-PSU cells in every "
                                         "training fold, so a row is predicted "
                                         "by a model that saw its cluster-mates",
                            "magnitude_source": "ablation.json "
                                                "cluster_holdout, which holds "
                                                "PSUs out instead of rows",
                            # Read, not re-typed: see the note on the helper.
                            **cluster_optimism_from_ablation((hi - lo) / 2.0),
                            "direction": "the accuracy below is optimistic by "
                                         "this much",
                            "why_not_fixed": "a PSU-held-out estimate cannot "
                                             "carry a design-based interval on "
                                             "the full design, which is the "
                                             "point of this block, so the "
                                             "optimism is disclosed rather "
                                             "than removed. Read "
                                             "exceeds_ci_half_width before "
                                             "treating it as small: this key "
                                             "used to end 'the optimism is "
                                             "smaller than the interval', and "
                                             "the worst rung's optimism is "
                                             "about 1.1x this interval's "
                                             "half-width, so that clause was "
                                             "false in the same dict as the "
                                             "number that refutes it",
                        },
                        "metrics": m}

    print(f"  {'metric':<20}{'unwtd':>9}{'weighted':>10}{'SE':>8}"
          f"{'95% CI':>20}{'unwtd inside':>14}")
    inside = 0
    for nm, r in m.items():
        if r.get("undefined"):
            print(f"  {nm:<20}{'undefined':>9}   {r['reason']}")
            continue
        lo, hi = r["ci95"] if nm == "pr_auc" else r["ci95_logit"]
        ok = bool(lo <= r["prop_unweighted"] <= hi)
        inside += ok
        r["unweighted_inside_ci"] = ok
        print(f"  {nm:<20}{r['prop_unweighted']:9.4f}{r['prop_weighted']:10.4f}"
              f"{r['se']:8.4f}   [{lo:.4f},{hi:.4f}]"
              f"{'yes' if ok else 'NO':>14}")
    n_def = sum(1 for r in m.values() if not r.get("undefined"))
    out["model_oof"]["unweighted_inside_ci"] = f"{inside} of {n_def}"
    print(f"  {inside} of {n_def} unweighted values fall inside the population "
          f"interval. For the")
    print("  ones that do not, weighting is not cosmetic: those are the numbers "
          "a")
    print("  sample-only paper reports wrongly for the population it claims.")
    print(f"  pr_auc uses a Rao-Wu rescaled bootstrap on "
          f"{m['pr_auc']['replicates']} replicates; the other four")
    print("  use the same linearised ratio estimator as the prevalence table.")

    maj_un = float(max(y.mean(), 1 - y.mean()))
    pw = out["prevalence"]["Anemia"]["prop_weighted"]
    maj_wt = float(max(pw, 1 - pw))
    acc = m["accuracy"]
    out["model_oof"]["majority_baseline"] = {"unweighted": maj_un,
                                             "weighted": maj_wt}
    print(f"  {'majority baseline':<20}{maj_un:9.4f}{maj_wt:10.4f}")

    lift_un = acc["prop_unweighted"] - maj_un
    lift_wt = acc["prop_weighted"] - maj_wt
    out["model_oof"]["lift_over_baseline"] = {"unweighted": lift_un,
                                             "weighted": lift_wt}
    print(f"\n  lift over baseline   unweighted {lift_un*100:+.2f} points"
          f"   weighted {lift_wt*100:+.2f} points")
    print("  Both the baseline and the model move when the sample is weighted to")
    print("  the population, so the honest lift has to be quoted on one scale or")
    print("  the other - never a weighted model against an unweighted baseline.")

    # --------------------------------------- do the weights belong in the fit -
    print("\nWEIGHTS AT FITTING TIME, NOT ONLY AT SCORING TIME")
    print("  the same folds and the same forest, refitted with sample_weight = the")
    print("  survey weight, then scored on the population scale exactly as above")
    predw, probaw = oof_predictions(X, y, fit_weight=w)
    mw = svy_metrics(y, predw, w, st, ps)
    mw["pr_auc"] = svy_pr_auc(y, probaw, w, st, ps)

    tests = {
        "accuracy": lambda ww: (accuracy_score(y, predw, sample_weight=ww)
                                - accuracy_score(y, pred, sample_weight=ww)),
        "sensitivity_recall": lambda ww: (
            recall_score(y, predw, sample_weight=ww)
            - recall_score(y, pred, sample_weight=ww)),
        "pr_auc": lambda ww: (
            average_precision_score(y, probaw, sample_weight=ww)
            - average_precision_score(y, proba, sample_weight=ww)),
    }
    contrasts = {k: paired_design_test(f, st, ps, w) for k, f in tests.items()}

    # Holm across the three, because they are one family asked one question:
    # "does moving the weights into the fit change the population-scale model?"
    # This file used to read c["p_value"] raw while ablation.py corrected its six
    # contrasts and benchmark.py corrected its family - the same three scripts,
    # two of which had a hand-written `holm` and this one of which had none. The
    # correction is not cosmetic: it roughly triples the smallest of the three and
    # doubles the middle one, and every threshold comparison below now reads the
    # adjusted value. Only accuracy survives it, and it is the only one that was
    # ever close.
    #
    # The sizes are described rather than quoted. This comment used to carry six
    # decimals of raw-and-adjusted p, and by the time anyone read it two of the
    # six were wrong - the numbers had moved with the replicate count and the
    # comment had not. Both values are in the artefact under p_value and p_holm,
    # which is where a reader should be reading them from anyway.
    fam = list(contrasts)
    for k, adj in zip(fam, dp.holm([contrasts[k]["p_value"] for k in fam])):
        contrasts[k]["p_holm"] = adj
        contrasts[k]["significant_holm_05"] = bool(adj < 0.05)
    out["fit_weight_sensitivity"] = {
        "weights_used_in_fit": True,
        "metrics_population_scale": mw,
        "paired_contrasts_weighted_minus_unweighted_fit": contrasts,
        "significance": {
            "family": fam,
            "correction": "Holm step-down over the three contrasts above",
            "note": ("p_holm is the value every verdict in this block reads. "
                     "p_value is kept beside it so a reader can see the size of "
                     "the correction rather than take it on trust."),
        },
        "n_predictions_changed": int((predw != pred).sum()),
        "pct_predictions_changed": round(float((predw != pred).mean() * 100), 3),
    }
    print(f"  {'metric (population scale)':<26}{'unwtd fit':>10}{'wtd fit':>9}"
          f"{'delta':>9}{'95% CI':>19}{'p':>7}{'p Holm':>8}")
    for nm in tests:
        c = contrasts[nm]
        lo, hi = c["ci95"]
        print(f"  {nm:<26}{m[nm]['prop_weighted']:10.4f}"
              f"{mw[nm]['prop_weighted']:9.4f}{c['delta']:+9.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]{c['p_value']:7.3f}"
              f"{c['p_holm']:8.3f}")
    ch = out["fit_weight_sensitivity"]
    print(f"  the two models disagree on {ch['n_predictions_changed']:,} of "
          f"{len(y):,} rows ({ch['pct_predictions_changed']}%)")
    print("  p Holm is the corrected column and the one the verdict uses; three "
          "contrasts, one family.")
    moved = [k for k, c in contrasts.items() if c["significant_holm_05"]]
    out["fit_weight_sensitivity"]["metrics_that_move_at_05"] = moved
    if moved:
        d = ", ".join(f"{k} {contrasts[k]['delta']:+.4f} "
                      f"(Holm p {contrasts[k]['p_holm']:.3f})" for k in moved)
        helped = [k for k in moved if contrasts[k]["delta"] > 0]
        out["fit_weight_sensitivity"]["verdict"] = (
            f"Fitting with survey weights changes {ch['pct_predictions_changed']}% "
            f"of predictions and moves {len(moved)} of {len(contrasts)} "
            f"population-scale metrics detectably after Holm correction across "
            f"the three ({d}). "
            + ("It improves the ones that move, so the unweighted fit is a "
               "reporting choice with a measured cost."
               if len(helped) == len(moved) else
               "It does not improve them, so weighting the fit is not a free "
               "upgrade: the unweighted fit is reported because it is what the "
               "manuscript trained AND because weighting the fit does not help "
               "on the population scale."))
        print(f"  {len(moved)} of {len(contrasts)} metrics move detectably after "
              f"Holm: {d}.")
        print("  So the design does not stop at reporting - it changes the "
              "fitted model.")
        if len(helped) != len(moved):
            print("  Note the sign: weighting the fit does NOT improve the "
                  "population-scale")
            print("  numbers here. Survey weights make the estimator "
                  "unbiased for population")
            print("  QUANTITIES; they do not make a classifier better at "
                  "classifying, because")
            print("  they upweight rows for representativeness, not for "
                  "difficulty. So the")
            print("  reported model stays the unweighted fit - both because it "
                  "is what the")
            print("  manuscript trained and because the weighted fit is not an "
                  "improvement.")
        else:
            print("  The weighted fit is the better one on the population "
                  "scale; the reported")
            print("  model stays the unweighted fit for comparability with the "
                  "ladder, and")
            print("  this table is the disclosure of what that costs.")
    else:
        out["fit_weight_sensitivity"]["verdict"] = (
            f"No population-scale metric moves at Holm-adjusted 0.05 on "
            f"{contrasts['accuracy']['df']} df, so weighting the fit is a "
            f"measured null. Weighting the scoring is not.")
        print(f"  none of the {len(contrasts)} metrics moves detectably (all "
              f"Holm p >= 0.05) on {contrasts['accuracy']['df']} df,")
        print("  so weighting the fit is a defensible thing to skip - stated as "
              "a measured")
        print("  null rather than left as an unexamined choice. Weighting the "
              "SCORING,")
        print("  above, is not optional.")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
