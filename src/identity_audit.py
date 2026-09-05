"""
Quantify the algebraic redundancy of the NHANES red-cell block.

The claim to be tested: the six routine red-cell indices
    {RBC, HGB, HCT, MCV, MCH, MCHC}
are linked by three exact analyser identities

    MCV  = HCT / RBC * 10        (fL)
    MCH  = HGB / RBC * 10        (pg)
    MCHC = HGB / HCT * 100       (g/dL)

so the block carries only THREE free measurements, not six, and hemoglobin is
recoverable from several sets that do not contain it:

    HGB = MCH  * RBC  / 10
    HGB = MCHC * HCT  / 100
    HGB = MCHC * MCV  * RBC / 1000
    HGB = MCH  * HCT  / MCV

and hematocrit likewise:

    HCT = MCV  * RBC  / 10

The fourth hemoglobin route is the one that is easy to miss: it uses no RBC and
no MCHC, so a feature set stripped of both still carries the label.

Deleting HGB from a feature matrix that retains any such set therefore does not
remove the label from the inputs.

Five things here are answers to obvious objections.

  The three-degrees-of-freedom claim is DEFINITIONAL, not a rank estimate: six
  published variables minus three exact constraints. The rank numbers below are
  the empirical check on it, and they are not the claim. "Rank 3" holds in LOGS,
  where the identities are linear. In levels they are products, so a linear rank
  test on the raw block reports one more, and quoting that as if it contradicted
  the identities is a category error. Both are printed at the same
  cumulative-variance cutoff, next to each other, with the reason. The stable
  finding is the gap of one, not the absolute rank, which moves with the cutoff.
  An earlier version of section 5 printed that gap of one under the label "the
  identities cost N dimension(s)", which read as a claim that the identities
  remove one dimension of six when the whole paper argues they remove three.

  The identities do not hold to machine precision, because NHANES publishes
  these variables rounded. Rather than wave that away, each identity is checked
  against a per-row bound propagated from the reporting precision, and the rows
  that exceed it are counted and shown: those are data errors, not rounding.

  The rounding null imitates the whole reporting chain, which rounds the three
  measured variables AND the three derived ones. Building the null from the
  already-rounded inputs makes it several times too small; the factor is
  computed and reported as naive_understates_null_by, not asserted in prose.

  Label agreement uses each person's own WHO cut-off, not a flat 12.0 g/dL.

  The named routes are not asserted to be the only ones. Every non-empty subset
  of the five non-hemoglobin indices is searched, so the claim "these are the
  exact routes" is a result rather than an assumption. The four named routes are
  also cross-checked against dataprep.HGB_ROUTES, which is the list the ladder's
  cleanliness tagger uses, so the two cannot drift apart unnoticed.

Outputs
  identity_audit.json   machine-readable results
  stdout                the table to quote in the paper

Run:  python identity_audit.py
"""

import json
import sys
from itertools import combinations

import numpy as np

import dataprep as dp
from download_data import CYCLES, PRIMARY
from paths import RESULTS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = RESULTS / "identity_audit.json"

# short name -> NHANES column, and the reverse for display
BLOCK = {"RBC": dp.RBC, "HGB": dp.HGB, "HCT": dp.HCT,
         "MCV": dp.MCV, "MCH": dp.MCH, "MCHC": dp.MCHC}
SHORT = {v: k for k, v in BLOCK.items()}

# Number of published decimals, and the half-width of that grid. These are the
# same fact twice, so DECIMALS used to sit 130 lines below HALF with nothing
# tying them together: editing one and not the other would have changed the
# rounding bounds in section 1 without changing the null in section 5, or the
# reverse. They are adjacent and checked now.
DECIMALS = {"RBC": 2, "HGB": 1, "HCT": 1, "MCV": 1, "MCH": 1, "MCHC": 1}
HALF = {k: 0.5 * 10.0 ** -d for k, d in DECIMALS.items()}
assert HALF == {"RBC": 0.005, "HGB": 0.05, "HCT": 0.05,
                "MCV": 0.05, "MCH": 0.05, "MCHC": 0.05}, HALF

# The "exact" cutoff for a reconstruction, in g/dL. HGB is published to one
# decimal, so 0.05 is the half-width of its own grid: a route that lands inside
# it is indistinguishable from the published value at the precision NHANES
# reports. It is NOT an estimate of the achievable error floor - uniform
# 1-decimal rounding alone gives MAE about 0.025 - and the result does not
# depend on the choice, because subset_search reports the gap between the worst
# exact subset and the best non-exact one. Any cutoff inside that gap gives the
# same answer.
EXACT_MAE = HALF["HGB"]

# Rounding for the JSON leaves. SVD and lstsq results differ in the last bits
# between BLAS builds, which would move the run_all manifest hash without any
# published figure changing; nothing here is printed past 6 decimals.
ND = 10

# How many rows of the subset table to print. It was the literal 8 in two
# places, once in the slice and once in the "... N more" arithmetic, so
# changing one would have made the count wrong.
TOP_N = 8


def _r(x, nd=ND):
    """Round a float, or a list of them, for the JSON."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [round(float(v), nd) for v in x]
    return round(float(x), nd)


def block(df):
    """The six indices as a dict of float arrays, short names."""
    return {k: df[v].to_numpy(dtype=float) for k, v in BLOCK.items()}


def err_stats(truth, est, cuts=None):
    """Error summary. cuts = per-row WHO cut-off for label agreement."""
    d = est - truth
    out = {"mae": _r(np.abs(d).mean()),
           "rmse": _r(np.sqrt((d ** 2).mean())),
           "max_abs": _r(np.abs(d).max()),
           "bias": _r(d.mean()),
           "pearson_r": _r(np.corrcoef(truth, est)[0, 1])}
    if cuts is not None:
        out["label_agreement_who"] = _r(((truth < cuts) == (est < cuts)).mean())
        out["disagreements"] = int(((truth < cuts) != (est < cuts)).sum())
    return out


def identity_check(name, reported, recomputed, tol, half_reported):
    """
    One identity, checked against the precision it is actually published at.

    tol is the per-row bound propagated from the reporting grid of the inputs;
    half_reported is the grid of the reported value itself. A row is a genuine
    violation only if it misses by more than the two together.
    """
    d = np.abs(recomputed - reported)
    bound = tol + half_reported
    bad = d > bound
    return {"identity": name,
            "mae": _r(d.mean()),
            "rmse": _r(np.sqrt((d ** 2).mean())),
            "max_abs": _r(d.max()),
            "median_rounding_bound": _r(np.median(bound)),
            "rows_within_rounding": int((~bad).sum()),
            "rows_exceeding": int(bad.sum()),
            "pct_exceeding": round(float(bad.mean() * 100), 4),
            "worst_row_excess": _r((d - bound).max()),
            "worst_row_excess_note": "by how much the worst row misses its own "
                                     "bound. Positive is a genuine violation; "
                                     "with no violations it is negative and is "
                                     "the tightest margin, not an excess",
            "pearson_r": _r(np.corrcoef(reported, recomputed)[0, 1])}


def _identity_defs(b):
    """
    The three identities as data: (label, reported key, recomputed, input bound).

    identity_check() and violation_mask() each used to carry their own copy of
    these three formulas and their three propagated bounds - six expressions
    where there are three. Editing one pair and not the other would have changed
    section 1's violation counts without changing which rows section 5 excludes,
    or the reverse, with nothing to catch it. There is one copy now.

    Each bound is the first-order propagation of the inputs' own grid half-widths
    through the formula; identity_check adds the grid of the reported value.
    """
    rbc, hgb, hct = b["RBC"], b["HGB"], b["HCT"]
    return [
        ("MCV = HCT/RBC*10", "MCV", hct / rbc * 10.0,
         10.0 * (HALF["HCT"] / rbc + hct * HALF["RBC"] / rbc ** 2)),
        ("MCH = HGB/RBC*10", "MCH", hgb / rbc * 10.0,
         10.0 * (HALF["HGB"] / rbc + hgb * HALF["RBC"] / rbc ** 2)),
        ("MCHC = HGB/HCT*100", "MCHC", hgb / hct * 100.0,
         100.0 * (HALF["HGB"] / hct + hgb * HALF["HCT"] / hct ** 2)),
    ]


N_IDENTITIES = 3        # checked against _identity_defs on every call below


def identities(b):
    """The three defining identities with propagated rounding bounds."""
    defs = _identity_defs(b)
    assert len(defs) == N_IDENTITIES, len(defs)
    return [identity_check(name, b[key], rec, tol, HALF[key])
            for name, key, rec, tol in defs]


def exact_routes(b, cuts):
    """
    The named closed-form routes to hemoglobin, and the one to hematocrit.

    Four routes to hemoglobin, not three. The fourth, MCH*HCT/MCV, uses no RBC
    at all: it is MCH*RBC/10 with RBC replaced by HCT/MCV*10, which is the MCV
    identity rearranged. It matters because a feature set can be stripped of
    both RBC and MCHC and still carry hemoglobin. The exhaustive search in
    subset_search finds all of them without being told; this dict is only the
    named, closed-form list.

    That list has to agree with dataprep.HGB_ROUTES, because HGB_ROUTES is what
    the ladder's cleanliness tagger tests feature sets against: a route named
    here and missing there would mark a leaky rung clean. The docstring used to
    say the two were checked against each other while the routes lived here only
    as whitespace-padded display strings with nothing machine-comparable in
    them. Each route now carries the column set it needs, and the assert below
    is the check - the same one ablation.py already makes.
    """
    hgb, hct = b["HGB"], b["HCT"]
    to_hgb = [
        ("MCH*RBC/10", {dp.MCH, dp.RBC},
         b["MCH"] * b["RBC"] / 10.0),
        ("MCHC*HCT/100", {dp.MCHC, dp.HCT},
         b["MCHC"] * b["HCT"] / 100.0),
        ("MCHC*MCV*RBC/1000", {dp.MCHC, dp.MCV, dp.RBC},
         b["MCHC"] * b["MCV"] * b["RBC"] / 1000.0),
        ("MCH*HCT/MCV", {dp.MCH, dp.HCT, dp.MCV},
         b["MCH"] * b["HCT"] / b["MCV"]),
    ]
    here = {frozenset(need) for _, need, _ in to_hgb}
    there = {frozenset(r) for r in dp.HGB_ROUTES}
    assert here == there, (
        "identity_audit's named routes and dataprep.HGB_ROUTES disagree; the "
        "ladder's leakage tagging is only as complete as HGB_ROUTES.\n"
        f"  only here:  {sorted(sorted(s) for s in here - there)}\n"
        f"  only there: {sorted(sorted(s) for s in there - here)}")

    out = {"to_hgb": {}, "cross_checked_against": "dataprep.HGB_ROUTES"}
    for label, need, est in to_hgb:
        s = err_stats(hgb, est, cuts)
        s["needs"] = sorted(SHORT[c] for c in need)
        s["needs_columns"] = sorted(need)
        out["to_hgb"][label] = s

    s = err_stats(hct, b["MCV"] * b["RBC"] / 10.0)
    s["needs"] = ["MCV", "RBC"]
    s["needs_columns"] = sorted([dp.MCV, dp.RBC])
    out["to_hct"] = {"MCV*RBC/10": s}
    return out


def subset_search(b, cuts):
    """
    Every subset of the five non-hemoglobin indices, scored on how well it
    rebuilds hemoglobin.

    The model is a log-linear fit, log HGB ~ sum a_j log x_j, because every
    analyser identity is a product and therefore exactly linear in logs. A
    subset that contains a genuine route recovers a coefficient vector of small
    integers and an error at the rounding floor; a subset that does not lands
    orders of magnitude worse. Nothing here is a guess about which sets matter.

    Two honest qualifications, both now reported in the returned dict rather
    than left for a reviewer to raise.

      These are IN-SAMPLE fits: the coefficients are least-squares estimates on
      all n rows and the MAE is evaluated on those same rows. With n above ten
      thousand and at most six parameters the optimism is negligible, but it is
      not zero, and it is worth being explicit that the four named routes in
      exact_routes are closed forms with no fitted parameter at all - those need
      no such caveat.

      The exact/non-exact split at EXACT_MAE is not a knife edge. The worst
      subset called exact and the best subset called non-exact are reported with
      the gap between them, so a reader can see that the classification is
      stable over any cutoff inside that gap rather than an artefact of 0.05.

    Returns rows sorted by MAE, and the smallest exact sets found.
    """
    names = [k for k in BLOCK if k != "HGB"]
    y = np.log(b["HGB"])
    hgb = b["HGB"]
    rows = []
    for r in range(1, len(names) + 1):
        for combo in combinations(names, r):
            X = np.column_stack([np.log(b[c]) for c in combo]
                                + [np.ones_like(y)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            est = np.exp(X @ coef)
            d = np.abs(est - hgb)
            rows.append({
                "features": list(combo),
                "k": r,
                "mae": _r(d.mean()),
                "max_abs": _r(d.max()),
                "exponents": [round(float(c), 4) for c in coef[:-1]],
                "intercept": round(float(coef[-1]), 4),
                "label_agreement_who": _r(((hgb < cuts) == (est < cuts)).mean()),
            })
    rows.sort(key=lambda r_: r_["mae"])

    exact = [r_ for r_ in rows if r_["mae"] < EXACT_MAE]
    non_exact = [r_ for r_ in rows if r_["mae"] >= EXACT_MAE]
    # Two subsets tie at k=2 ({MCH,RBC} and {MCHC,HCT}), so reporting one as
    # "the smallest" implied a uniqueness that does not hold. `minimal_exact`
    # lists all of them and is the field to read. `smallest_exact` is kept
    # because the cross-cycle `replication` block in main() copies it as that
    # cycle's one-line summary, and a list of ties would not fit there. The
    # comment here used to say that block PRINTS it; it does not - the replication
    # table prints the two named route MAEs - so the only reader of this field is
    # the JSON, and the field means the best of the minimal ones, not the only one.
    k_min = min((r_["k"] for r_ in exact), default=None)
    minimal = [r_ for r_ in exact if r_["k"] == k_min]
    return {"n_subsets": len(rows), "rows": rows,
            "exact_threshold_mae": EXACT_MAE,
            "exact_threshold_note":
                "half the 0.1 g/dL grid HGB is published on, not an estimate of "
                "the error floor (pure 1-decimal rounding gives MAE ~0.025)",
            "n_exact": len(exact),
            "worst_exact_mae": exact[-1]["mae"] if exact else None,
            "best_non_exact_mae": non_exact[0]["mae"] if non_exact else None,
            "gap_at_threshold": (_r(non_exact[0]["mae"] - exact[-1]["mae"])
                                 if exact and non_exact else None),
            "gap_note": "any cutoff inside this gap gives the same n_exact, so "
                        "the split does not depend on the 0.05 choice",
            "fit_note": "log-linear coefficients are fitted by least squares on "
                        "all n rows and scored on the same rows; the named "
                        "routes in `routes` are closed forms with no fitted "
                        "parameter and carry no such caveat",
            "minimal_exact": minimal,
            "smallest_exact": (min(minimal, key=lambda r_: r_["mae"])
                               if minimal else None)}


def rounding_null(b, n_draws=25, seed=dp.SEED):
    """
    What the log spectrum WOULD look like if the identities held exactly.

    The point of comparison has to imitate the whole reporting chain, and the
    chain rounds three times, not once. The analyser holds RBC, HGB and HCT at
    full precision, computes the three derived indices from those unrounded
    internals, and only then is every one of the six published on a grid. So the
    null is built by de-rounding: each published RBC, HGB and HCT is replaced by
    a value drawn uniformly inside its own grid cell, the derived indices are
    computed exactly from those, and all six are rounded.

    Deriving the null from the already-rounded RBC, HGB and HCT instead - the
    obvious thing to do, and what an earlier version of this function did - is
    wrong in a way that flatters the result. It lets the derived indices inherit
    exactly the rounding error of their inputs, so the trailing variance of the
    null comes out several times too small and the observed data looks that much
    more anomalous than it is. Both versions are reported below, along with the
    ratio between them: the factor used to be quoted in prose as "about 2.8x",
    which is a number no reader could check and which would silently go stale
    the first time the cohort changed.

    If the trailing variance of the null matches the trailing variance of the
    real data, the real block is rank 3 and what is left over is the grid.

    The draw is averaged over n_draws de-rounding samples because a single draw
    is itself noisy; the spread across draws is reported so the ratio is never
    read as more precise than it is.

    n_draws=25 and seed=dp.SEED are the published settings: no caller overrides
    either, and both are echoed into the artefact so the numbers in the paper can
    be reproduced without reading this signature. They are parameters rather
    than literals only so a reader can re-run the null at a different draw count
    and see the sd across draws move; nothing in the pipeline does.
    """
    rbc, hgb, hct = b["RBC"], b["HGB"], b["HCT"]

    def spectrum_of(cols):
        M = np.column_stack([np.round(cols[k], DECIMALS[k]) for k in BLOCK])
        L = np.log(M)
        L = (L - L.mean(0)) / L.std(0, ddof=0)
        sv = np.linalg.svd(L, compute_uv=False)
        var = sv ** 2 / (sv ** 2).sum()
        return var

    # the naive construction, kept only to show what it costs
    naive = spectrum_of({"RBC": rbc, "HGB": hgb, "HCT": hct,
                         "MCV": hct / rbc * 10.0,
                         "MCH": hgb / rbc * 10.0,
                         "MCHC": hgb / hct * 100.0})

    rng = np.random.default_rng(seed)
    trails, vars_ = [], []
    for _ in range(n_draws):
        def jitter(x, key):
            h = HALF[key]
            return x + rng.uniform(-h, h, size=len(x))
        r_, g_, c_ = jitter(rbc, "RBC"), jitter(hgb, "HGB"), jitter(hct, "HCT")
        var = spectrum_of({"RBC": r_, "HGB": g_, "HCT": c_,
                           "MCV": c_ / r_ * 10.0,
                           "MCH": g_ / r_ * 10.0,
                           "MCHC": g_ / c_ * 100.0})
        vars_.append(var)
        trails.append(float(var[3:].sum() * 100))

    var_mean = np.mean(vars_, axis=0)
    trail = float(np.mean(trails))
    naive_trail = float(naive[3:].sum() * 100)
    return {"n_draws": int(n_draws),
            # Published, not just used: the de-rounding draw is random, so a
            # trailing variance quoted without its seed is not reproducible.
            "seed": int(seed),
            "n_rows": int(len(rbc)),
            "variance_ratio": _r(var_mean),
            "trailing3_variance_pct": _r(trail),
            "trailing3_variance_pct_sd_across_draws": _r(np.std(trails)),
            "naive_derived_from_rounded_inputs": {
                "variance_ratio": _r(naive),
                "trailing3_variance_pct": _r(naive_trail),
                "why_not_used": "the derived indices inherit their inputs' "
                                "rounding error, so this understates the null "
                                "and overstates the anomaly"},
            "naive_understates_null_by": _r(trail / max(naive_trail, 1e-12), 3)}


def violation_mask(b):
    """
    Rows whose own reported indices are mutually inconsistent by more than the
    reporting grid allows, on any of the three identities.

    The per-identity rates are in identities(); the union is what matters, and
    it is reported as clean_rows.pct_excluded rather than quoted here, because a
    prose range ("about 0.6-0.9% on any one identity") went stale as soon as the
    cohort changed and had already stopped covering the worst identity.

    These rows matter twice: they are the reason the trailing log variance is
    larger than pure rounding would give, and they are a reminder that these are
    real analyser outputs with real transcription and instrument errors in them.

    The bounds come from _identity_defs, the same source section 1 uses, so a
    row counted as a violation in the table is exactly a row excluded here.
    """
    bad = np.zeros(len(b["RBC"]), dtype=bool)
    for _, key, rec, tol in _identity_defs(b):
        bad |= np.abs(rec - b[key]) > tol + HALF[key]
    return bad


def ranks(b, cum_frac=0.999):
    """
    Rank of the block in levels and in logs, and why they differ.

    The three-degrees-of-freedom claim is arithmetic, not statistical: six
    published variables, three exact constraints, three free measurements. That
    is `degrees_of_freedom` below and it does not depend on any cutoff. What the
    spectra add is the check that the constraints really are exact in this data.

    In logs the three identities are three exact linear constraints, so the
    standardised log block carries its variance in three directions and the last
    three hold nothing but rounding. In levels the identities are products, so a
    linear rank test cannot see them and reports one more. Reporting only the
    levels number would understate the redundancy; reporting only the log number
    without this paragraph invites the objection that the raw block "looks like
    rank 4".

    Both ranks are taken at the SAME cumulative-variance cutoff, because the
    whole point is a comparison, and both are also reported at 99.9% and 99.99%
    so the cutoff-dependence is visible instead of asserted: the gap of one is
    the stable finding, the absolute number is not.

    cum_frac=0.999 is the published cutoff and no caller overrides it; it is
    echoed into the artefact as `cumulative_variance_cutoff` so the component
    counts below are never read without it. Nothing here should be quoted as the
    degrees of freedom - that is the arithmetic `degrees_of_freedom` key, which
    the two component counts are deliberately named apart from.

    Note that the spectra are of the STANDARDISED matrix - each column centred
    and divided by its own SD, i.e. a correlation-matrix PCA - not merely the
    centred one. Without standardising, HCT (tens of %) and RBC (units of
    10^6/uL) would enter on wildly different scales and PC1 would mostly be a
    statement about units.

    rounding_null gives the comparison that decides whether the trailing three
    log directions are rounding or a fourth measurement, and the clean-row
    figure below has a control attached, because excluding the rows with the
    largest identity residuals and then measuring the residual variance selects
    on the very quantity it reports. Dropping the same NUMBER of rows at random
    says how much of the drop is that selection and how much is real.
    """
    names = list(BLOCK)
    M = np.column_stack([b[k] for k in names])

    def spectrum(A):
        A = (A - A.mean(0)) / A.std(0, ddof=0)
        sv = np.linalg.svd(A, compute_uv=False)
        var = sv ** 2 / (sv ** 2).sum()
        return sv, var

    sv_lvl, var_lvl = spectrum(M)
    sv_log, var_log = spectrum(np.log(M))
    null = rounding_null(b)

    # the same trailing variance with the mutually inconsistent records out,
    # against a null rebuilt on those same rows and a random-exclusion control
    bad = violation_mask(b)
    _, var_clean = spectrum(np.log(M[~bad]))
    null_clean = rounding_null({k: v[~bad] for k, v in b.items()})

    rng = np.random.default_rng(dp.SEED + 1)
    keep_rand = np.ones(len(bad), dtype=bool)
    keep_rand[rng.choice(len(bad), int(bad.sum()), replace=False)] = False
    _, var_rand = spectrum(np.log(M[keep_rand]))

    def k_at(var, frac):
        return int(np.searchsorted(np.cumsum(var), frac) + 1)

    obs_trail = float(var_log[3:].sum() * 100)
    clean_trail = float(var_clean[3:].sum() * 100)
    rand_trail = float(var_rand[3:].sum() * 100)
    null_trail = null["trailing3_variance_pct"]
    null_clean_trail = null_clean["trailing3_variance_pct"]
    k_log = k_at(var_log, cum_frac)
    k_lvl = k_at(var_lvl, cum_frac)
    return {"variables": names,
            "cumulative_variance_cutoff": cum_frac,
            "standardisation": "each column centred and divided by its own SD "
                               "(correlation-matrix PCA), in levels and in logs",
            "levels": {"singular_values": _r(sv_lvl, 6),
                       "variance_ratio": _r(var_lvl),
                       "k": k_lvl,
                       "k_999": k_at(var_lvl, 0.999),
                       "k_9999": k_at(var_lvl, 0.9999)},
            "logs": {"singular_values": _r(sv_log, 6),
                     "variance_ratio": _r(var_log),
                     "k": k_log,
                     "k_999": k_at(var_log, 0.999),
                     "k_9999": k_at(var_log, 0.9999),
                     "trailing3_variance_pct": _r(obs_trail)},
            "k_note": "`k` is the count at cumulative_variance_cutoff, so it "
                      "equals k_999 while that cutoff is 0.999; k_9999 is there "
                      "to show the count moves with the cutoff and the gap "
                      "between levels and logs does not",
            "rounding_null": null,
            "clean_rows": {"n_excluded": int(bad.sum()),
                           "pct_excluded": round(float(bad.mean() * 100), 3),
                           "variance_ratio": _r(var_clean),
                           "trailing3_variance_pct": _r(clean_trail),
                           "row_matched_null_trailing3_variance_pct":
                               _r(null_clean_trail),
                           "random_exclusion_control": {
                               "n_excluded": int(bad.sum()),
                               "trailing3_variance_pct": _r(rand_trail),
                               "what_it_is": "the same number of rows dropped "
                                             "at random instead of by residual; "
                                             "the distance from this to the "
                                             "clean figure is what the "
                                             "exclusion actually bought"}},
            "trailing_ratio_observed_to_null": round(
                obs_trail / max(null_trail, 1e-12), 3),
            "trailing_ratio_clean_to_null": round(
                clean_trail / max(null_clean_trail, 1e-12), 3),
            "trailing_ratio_random_to_null": round(
                rand_trail / max(null_trail, 1e-12), 3),
            "ratio_note": "the clean ratio uses a null rebuilt on the surviving "
                          "rows, not the full-cohort null, so numerator and "
                          "denominator describe the same rows",
            "n_identities": N_IDENTITIES,
            "degrees_of_freedom": len(names) - N_IDENTITIES,
            # No `log_rank_at_cutoff` here. It was a third copy of k_log, which
            # is already published as logs.k and, at this cutoff, again as
            # logs.k_999 - written by this function, read by nothing, and named
            # "rank" for a quantity that is a variance-truncation count. Two
            # copies with a note explaining why they coincide is the most this
            # number needs.
            "levels_minus_logs": k_lvl - k_log,
            "note": f"degrees_of_freedom is arithmetic: {len(names)} published "
                    f"variables minus {N_IDENTITIES} exact identities. The "
                    f"spectra are the check, not the claim: at {cum_frac:.4g} "
                    f"cumulative variance the standardised log block needs "
                    f"{k_log} components and a levels test at the same cutoff "
                    f"needs {k_lvl - k_log} more, because in levels the "
                    f"identities are products and a linear rank test cannot see "
                    f"them. That gap of {k_lvl - k_log} is a property of the "
                    f"rank test, not the cost of the identities"}


def audit_cycle(cycle):
    """
    The whole audit for one cycle. Cheap enough to run on all four.

    Returns the audit dict only. It used to return `(df, audit)`, and neither
    call site ever read the frame: main() unpacked it and then used the name
    nowhere, and the replication loop unpacked it into `_`. The tuple existed to
    be re-packed at the one site that already had the value.

    The positivity guard is not decoration: every formula in this file either
    takes a logarithm or divides by one of these six columns. A zero or a
    negative would have produced -inf, nan or a ZeroDivisionError-free garbage
    number that propagated all the way into the JSON, and json.dump would have
    written a bare `NaN` or `Infinity` token, which no strict JSON parser will
    read back. Failing here names the column instead.
    """
    df, _ = dp.load_cycle(cycle)
    b = block(df)
    bad_cols = {k: int((v <= 0).sum()) for k, v in b.items() if (v <= 0).any()}
    if bad_cols:
        raise SystemExit(f"identity_audit: {cycle} has non-positive values in "
                         f"{bad_cols}; every formula here logs or divides by "
                         f"these columns")
    nonfinite = {k: int((~np.isfinite(v)).sum())
                 for k, v in b.items() if not np.isfinite(v).all()}
    if nonfinite:
        raise SystemExit(f"identity_audit: {cycle} has non-finite values in "
                         f"{nonfinite}")
    cuts = df["WHO_NORMAL"].to_numpy(dtype=float)
    return {"cycle": cycle, "n": int(len(df)),
            "identities": identities(b),
            "routes": exact_routes(b, cuts),
            "subset_search": subset_search(b, cuts),
            "ranks": ranks(b)}


def main():
    out = {"primary_cycle": PRIMARY}
    prim = audit_cycle(PRIMARY)
    out["primary"] = prim
    n = prim["n"]
    print(f"NHANES {PRIMARY}: {n:,} rows in the shared cohort\n")

    print("1. ANALYSER IDENTITIES, against the precision NHANES publishes")
    print(f"   {'identity':<22}{'MAE':>8}{'max|e|':>9}"
          f"{'within rounding':>17}{'violations':>12}")
    for s in prim["identities"]:
        print(f"   {s['identity']:<22}{s['mae']:>8.4f}{s['max_abs']:>9.3f}"
              f"{s['rows_within_rounding']:>13,}/{n:<4,}"
              f"{s['rows_exceeding']:>10,} ({s['pct_exceeding']}%)")
    print("   a 'violation' misses by more than the reporting grid can explain,")
    print("   so it is a data error in that record, not evidence against the identity")

    print("\n2. HEMOGLOBIN RECONSTRUCTED FROM SETS THAT EXCLUDE IT")
    for name, s in prim["routes"]["to_hgb"].items():
        head = f"{name} (needs {'+'.join(s['needs'])})"
        print(f"   {head:42s} MAE={s['mae']:.4f} g/dL  r={s['pearson_r']:.6f}")
        print(f"   {'':42s} bias {s['bias']:+.4f} g/dL, "
              f"worst row {s['max_abs']:.3f}")
        print(f"   {'':42s} WHO label agreement "
              f"{s['label_agreement_who']*100:.2f}%  "
              f"({s['disagreements']:,} of {n:,} disagree)")
    print("   agreement uses each person's own WHO cut-off, not a flat 12.0 g/dL")
    print("   these four route definitions are asserted equal to "
          "dataprep.HGB_ROUTES,")
    print("   which is what the ladder's leakage tagging is built on")

    print("\n3. HEMATOCRIT RECONSTRUCTED THE SAME WAY")
    for name, s in prim["routes"]["to_hct"].items():
        head = f"{name} (needs {'+'.join(s['needs'])})"
        print(f"   {head:42s} MAE={s['mae']:.4f} %      r={s['pearson_r']:.6f}")
    hct_needs = set(prim["routes"]["to_hct"]["MCV*RBC/10"]["needs_columns"])
    carry = [k for k, f in dp.LADDER.items()
             if hct_needs <= set(f) and dp.HCT not in f]
    print("   this is why dropping HCT from the feature list does not remove it.")
    print(f"   rungs that omit HCT but keep {'+'.join(sorted(SHORT[c] for c in hct_needs))}"
          f", so still carry it implicitly:")
    print(f"     {', '.join(carry) if carry else 'none'}")
    print("   that is not leakage - hematocrit is not the label - but it is why")
    print("   the ladder is defined by routes to HEMOGLOBIN and not by which")
    print("   column names happen to be absent")

    ss = prim["subset_search"]
    print(f"\n4. EXHAUSTIVE SUBSET SEARCH  (all {ss['n_subsets']} subsets of the "
          f"{len(BLOCK) - 1} non-Hb indices)")
    print(f"   {'features':<28}{'k':>2}{'MAE g/dL':>11}{'WHO agree':>11}"
          f"   exponents")
    for r in ss["rows"][:TOP_N]:
        print(f"   {'+'.join(r['features']):<28}{r['k']:>2}{r['mae']:>11.4f}"
              f"{r['label_agreement_who']*100:>10.2f}%   {r['exponents']}")
    print(f"   ... {ss['n_subsets'] - TOP_N} more, worst "
          f"MAE {ss['rows'][-1]['mae']:.3f} g/dL "
          f"({'+'.join(ss['rows'][-1]['features'])})")
    print(f"   {ss['n_exact']} of {ss['n_subsets']} subsets reach the rounding "
          f"floor (MAE < {ss['exact_threshold_mae']} g/dL)")
    if ss["gap_at_threshold"] is not None:
        print(f"   that split is not a knife edge: worst exact "
              f"{ss['worst_exact_mae']:.4f} against best non-exact "
              f"{ss['best_non_exact_mae']:.4f} g/dL,")
        print(f"   a gap of {ss['gap_at_threshold']:.4f} g/dL, so any cutoff "
              f"inside it gives the same {ss['n_exact']}")
    minimal = ss["minimal_exact"]
    if minimal:
        print(f"   smallest exact routes ({minimal[0]['k']} variables each): "
              f"{', '.join('+'.join(m['features']) for m in minimal)}")
        for m in minimal:
            print(f"     {'+'.join(m['features']):<20} exponents "
                  f"{m['exponents']} -> MAE {m['mae']:.4f} g/dL")
    print("   exponents near +-1 are one identity being rediscovered by the fit.")
    top = ss["rows"][0]
    split = [e for e in top["exponents"] if 0.25 < abs(e) < 0.75]
    if len(split) >= 2:
        print("   the best row instead splits weight across two routes, which "
              "beats either")
        print("   single route because averaging cancels rounding noise")
    print("   these are least-squares fits on the same rows they are scored on, "
          "and")
    print("   near-collinear subsets can split one route's exponent between two")
    print("   columns, so read the MAE column as the result and the exponents as")
    print("   an illustration")

    print("\n5. RANK OF THE SIX-INDEX BLOCK")
    rk = prim["ranks"]
    cut = rk["cumulative_variance_cutoff"]
    print(f"   {'':6}{'levels':>12}{'logs':>14}{'logs, exact':>14}")
    for i in range(len(BLOCK)):
        print(f"   PC{i+1:<3}{rk['levels']['variance_ratio'][i]*100:>11.5f}%"
              f"{rk['logs']['variance_ratio'][i]*100:>13.6f}%"
              f"{rk['rounding_null']['variance_ratio'][i]*100:>13.6f}%")
    print(f"   -> degrees of freedom {rk['degrees_of_freedom']} of "
          f"{len(BLOCK)}: {len(BLOCK)} published variables minus "
          f"{rk['n_identities']} exact identities.")
    print("   that is arithmetic. everything below is the check on it.")
    print(f"   components for {cut*100:.4g}% of variance: levels "
          f"{rk['levels']['k']}, logs {rk['logs']['k']} "
          f"(at 99.99%: {rk['levels']['k_9999']} and "
          f"{rk['logs']['k_9999']})")
    print(f"   the levels test needs {rk['levels_minus_logs']} more than the log "
          f"test at either cutoff, because in levels")
    print("   the identities are products and a linear rank test cannot see "
          "them. that")
    print("   gap is a property of the rank test, not the cost of the identities")
    print("   'logs, exact' = the same six variables rebuilt from de-rounded RBC,")
    print("   HGB and HCT with no error and then rounded onto the published grid")
    print(f"   variance in the last three log directions, observed : "
          f"{rk['logs']['trailing3_variance_pct']:.4f}%")
    # The spread across de-rounding draws is ~5e-05 percentage points, so .4f
    # printed it as "+-0.0000" - a zero that reads as "no variability measured"
    # when what it means is "smaller than four decimals". Six decimals is the
    # width that shows the number instead of hiding it.
    print(f"   the same, for exact values rounded onto the grid    : "
          f"{rk['rounding_null']['trailing3_variance_pct']:.4f}% "
          f"(+-{rk['rounding_null']['trailing3_variance_pct_sd_across_draws']:.6f} "
          f"across {rk['rounding_null']['n_draws']} de-rounding draws)")
    print(f"   observed is {rk['trailing_ratio_observed_to_null']:.2f}x the "
          f"rounding-only figure")
    print("   building that null from the already-rounded indices instead would "
          "make it")
    print(f"   {rk['rounding_null']['naive_understates_null_by']:.2f}x smaller "
          f"and the anomaly look that much bigger")
    cr = rk["clean_rows"]
    print(f"   with the {cr['n_excluded']:,} mutually inconsistent records "
          f"({cr['pct_excluded']}%) dropped : "
          f"{cr['trailing3_variance_pct']:.4f}%,")
    print(f"   against a null rebuilt on those same rows "
          f"({cr['row_matched_null_trailing3_variance_pct']:.4f}%) -> "
          f"{rk['trailing_ratio_clean_to_null']:.2f}x")
    print(f"   control: dropping {cr['random_exclusion_control']['n_excluded']:,} "
          f"rows AT RANDOM instead leaves "
          f"{cr['random_exclusion_control']['trailing3_variance_pct']:.4f}% "
          f"({rk['trailing_ratio_random_to_null']:.2f}x),")
    print("   so the drop is the excluded rows and not the smaller sample. the")
    print("   exclusion selects on the residual it then measures, which is why "
          "the")
    print("   control is here rather than the clean figure standing alone.")
    print("   once those records are out the leftover matches rounding; it is not")
    print("   a fourth independent measurement.")

    # ---- replication ---------------------------------------------------------
    print("\n6. DO THE IDENTITIES HOLD IN ALL FOUR CYCLES?")
    out["replication"] = {}
    out["replication_note"] = (
        f"all four cycles, including the primary one ({PRIMARY}), so the "
        f"primary row is the same audit re-listed rather than a fifth dataset")
    print(f"   {'cycle':<12}{'n':>8}   "
          f"{'MCH route MAE':>14}{'MCHC route MAE':>16}{'WHO agree':>11}")
    for cycle in CYCLES:
        res = prim if cycle == PRIMARY else audit_cycle(cycle)
        out["replication"][cycle] = {
            "n": res["n"],
            "is_primary": cycle == PRIMARY,
            "identities": res["identities"],
            "routes": res["routes"],
            # Named for what they are: the number of principal components needed
            # to reach `cumulative_variance_cutoff`, in logs and in levels. They
            # were called log_dof and levels_dof, which is wrong twice over - a
            # variance-cutoff component count is not a rank and not a degree of
            # freedom, and this file's own `degrees_of_freedom` field is the
            # arithmetic 9 - 3 = 6 and disagrees with both numbers. Publishing
            # the component count under the dof name invited a reader to quote
            # a PCA truncation as the algebraic result.
            "log_components_at_cutoff": res["ranks"]["logs"]["k"],
            "levels_components_at_cutoff": res["ranks"]["levels"]["k"],
            "degrees_of_freedom": res["ranks"]["degrees_of_freedom"],
            "cumulative_variance_cutoff":
                res["ranks"]["cumulative_variance_cutoff"],
            "smallest_exact": res["subset_search"]["smallest_exact"],
        }
        r = res["routes"]["to_hgb"]
        mch = r["MCH*RBC/10"]
        mchc = r["MCHC*HCT/100"]
        star = " <- primary" if cycle == PRIMARY else ""
        print(f"   {cycle:<12}{res['n']:>8,}   {mch['mae']:>14.4f}"
              f"{mchc['mae']:>16.4f}{mch['label_agreement_who']*100:>10.2f}%{star}")
    print("   the leak is a property of the analyser, not of one survey cycle")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        # allow_nan=False: the default writes bare NaN and Infinity tokens, which
        # are not valid JSON and which several readers reject outright. A nan
        # reaching here would mean a division or a log went wrong upstream, and
        # the guard in audit_cycle should already have caught it - this is the
        # second line of defence, and it fails loudly instead of writing a file
        # that looks fine until something tries to parse it.
        json.dump(out, fh, indent=1, allow_nan=False)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
