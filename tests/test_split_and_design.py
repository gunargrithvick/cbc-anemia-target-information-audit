"""
The split and the survey design: the two things every reported number sits on.

The split is checked for the properties the paper claims of it - one split, shared
by every rung and every model, disjoint, exhaustive, deterministic, stratified on
Severity4 - and for the property it does NOT have, which is cluster separation.
That last one is asserted here as a fact about the split rather than treated as a
bug, because dataprep.split3's docstring says so out loud and ablation's
cluster_holdout block is what measures the cost.

The design side checks the degrees of freedom, that Rao-Wu replicate weights are
what they claim to be, and all five branches of dp.t_and_p, which exists because
three scripts used to answer the degenerate cases differently. It said "the three
branches" while the function documented five, and the two that were uncounted
were also the two untested: a non-finite input, which used to be published as
p = 0.0 with a false explanation attached, and zero residual degrees of freedom,
which returns NaN from stats.t.sf and breaks the "p_value is always a float"
promise every caller relies on.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_split_sizes_are_the_ones_the_paper_reports(splits):
    tr, va, te = splits
    assert (len(tr), len(va), len(te)) == (7293, 2431, 2432)


def test_split_is_disjoint_and_exhaustive(splits, cohort):
    tr, va, te = splits
    n = len(cohort[0])
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert set(tr) | set(va) | set(te) == set(range(n)), "the split loses rows"


def test_split_is_deterministic(dp, cohort):
    """Called twice, the same rows. Otherwise 'one shared split' means nothing."""
    a = dp.split3(cohort[0])
    b = dp.split3(cohort[0])
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_split_is_stratified_on_four_class_severity(splits, cohort):
    """
    Severe is 26 people. Left to chance a fold gets none of them.

    Checked as counts, not proportions: 16 / 5 / 5 is the whole severe class, and
    what matters is that none of the three blocks is empty of it, because a fold
    with zero severe cases makes the three-class head's macro recall undefined.
    """
    y = cohort[0]["Severity4"].to_numpy()
    full = np.bincount(y, minlength=4)
    assert full.tolist() == [10989, 840, 301, 26]
    for block, want in zip(splits, ([6592, 504, 181, 16],
                                    [2198, 168, 60, 5],
                                    [2199, 168, 60, 5])):
        assert np.bincount(y[block], minlength=4).tolist() == want
        assert np.bincount(y[block], minlength=4).min() > 0


def test_the_split_does_not_separate_survey_clusters(dp, cohort, splits):
    """
    Not a bug: a documented property, and the reason the paper reports contrasts.

    All 49 stratum x PSU cells appear in the test block, and all 49 appear in the
    training block too, so every test row comes from a cluster the model has
    already seen. If a future change made the split cluster-aware this test would
    fail, which is the point - it would also invalidate the optimism figures in
    ablation.json's cluster_holdout block, and both should be revisited together.
    """
    df = cohort[0]
    st = df[dp.STRATUM].to_numpy()
    ps = df[dp.PSU].to_numpy()
    cells = lambda i: {*zip(st[i].tolist(), ps[i].tolist())}
    everything = cells(np.arange(len(df)))
    assert len(everything) == 49
    tr, _, te = splits
    assert cells(te) == everything
    assert cells(tr) == everything


def test_design_degrees_of_freedom(dp, cohort):
    """49 PSU cells - 24 strata = 25. Every design-based interval uses this."""
    df = cohort[0]
    st, ps = df[dp.STRATUM].to_numpy(), df[dp.PSU].to_numpy()
    assert len(np.unique(st)) == 24
    assert len({*zip(st.tolist(), ps.tolist())}) == 49
    assert dp.design_df(st, ps) == 25


def test_rao_wu_weights_without_a_base_are_unweighted_replicates(dp, cohort):
    """
    base=None is the estimand switch: sample quantities, cluster-robust variance.

    The weights are per-row multipliers centred near 1, and inside a two-PSU
    stratum one PSU is scaled up and the other down, so a whole cluster can be
    zeroed out for a replicate - that is what makes the resulting SE
    cluster-robust rather than row-robust. What must NOT happen is MECWT leaking
    in: if it did, ablation.py and benchmark.py would be silently reporting
    population quantities under sample labels.
    """
    df = cohort[0]
    st, ps = df[dp.STRATUM].to_numpy(), df[dp.PSU].to_numpy()
    w = dp.rao_wu_weights(st, ps, np.random.default_rng(0), None)
    assert w.shape == (len(df),)
    assert w.min() >= 0.0
    assert 0.9 < float(w.mean()) < 1.1, float(w.mean())
    assert float(w.max()) > 1.0, "no cluster was ever scaled up"
    assert float(w.min()) == 0.0, "no cluster was ever dropped"
    mecwt = df[dp.MECWT].to_numpy()
    assert abs(float(w.mean()) - float(mecwt.mean())) > 1.0, "MECWT leaked into base=None"
    # A replicate weight is constant within a cluster - that is what "resample
    # PSUs, not rows" means.
    for cell in {*zip(st.tolist(), ps.tolist())}:
        m = (st == cell[0]) & (ps == cell[1])
        assert len(np.unique(w[m])) == 1, f"{cell} got row-varying weights"


def test_rao_wu_weights_with_a_base_stay_on_the_population_scale(dp, cohort):
    """base=MECWT multiplies the survey weight, so the total stays ~292 million."""
    df = cohort[0]
    st, ps = df[dp.STRATUM].to_numpy(), df[dp.PSU].to_numpy()
    mecwt = df[dp.MECWT].to_numpy()
    w = dp.rao_wu_weights(st, ps, np.random.default_rng(0), mecwt)
    assert 0.7 < float(w.sum()) / float(mecwt.sum()) < 1.4, float(w.sum())


def test_t_and_p_ordinary_case(dp):
    r = dp.t_and_p(0.01, 0.004, 25)
    assert r["t"] == pytest.approx(2.5)
    assert 0.0 < r["p_value"] < 0.05
    assert r["degenerate"] is None


def test_t_and_p_identical_predictions_is_p_one(dp):
    """0/0 is not infinite evidence. weighted.py used to say p = 0 here."""
    r = dp.t_and_p(0.0, 0.0, 25)
    assert r["t"] == 0.0
    assert r["p_value"] == 1.0
    assert r["degenerate"] and "identical" in r["degenerate"]


def test_t_and_p_unbounded_case_returns_no_statistic_and_stays_json_safe(dp):
    """
    A real difference no replicate moved: t is unbounded, so t is None, not inf.

    float('inf') is the failure this guards. Every script here dumps with
    allow_nan=False, so an infinite t would have crashed the write at the end of a
    long pipeline run instead of at the point of the problem, and p_value has to
    stay a float because callers Holm-correct it and compare it to 0.05.
    """
    import json
    r = dp.t_and_p(0.0148, 0.0, 25)
    assert r["t"] is None
    assert isinstance(r["p_value"], float) and r["p_value"] == 0.0
    assert r["degenerate"] and "unbounded" in r["degenerate"]
    json.dumps(r, allow_nan=False)


def test_t_and_p_non_finite_inputs_are_an_absent_test_not_a_significant_one(dp):
    """
    The branch whose absence published p = 0.0 as the study's strongest result.

    Before this case existed, a NaN fell through every guard - `se > 0` is False
    for NaN and `obs == 0` is False for NaN - and landed in the unbounded branch,
    which returns p = 0.0 with a `degenerate` string asserting a zero replicate
    spread. That is a specific false claim about what happened, attached to the
    smallest p-value in the file. Both inputs are checked, and the reason has to
    name which one was bad, or a reader cannot tell a broken delta from a broken
    SE.
    """
    import json
    import math
    for obs, se, word in ((math.nan, 0.004, "delta"),
                          (0.01, math.nan, "standard error"),
                          (math.inf, 0.004, "delta"),
                          (0.01, math.inf, "standard error")):
        r = dp.t_and_p(obs, se, 25)
        assert r["t"] is None, (obs, se)
        assert r["p_value"] == 1.0, (obs, se)
        assert r["degenerate"] and word in r["degenerate"], (obs, se)
        assert "not finite" in r["degenerate"], (obs, se)
        json.dumps(r, allow_nan=False)


def test_t_and_p_without_residual_degrees_of_freedom_declines_to_test(dp):
    """
    design_df returns 0 for a single-stratum subset, and stats.t.sf(x, 0) is NaN.

    The promise the docstring makes is that p_value is always a finite float,
    because every caller Holm-corrects it and compares it to 0.05. A NaN p
    silently loses that comparison and would be written by a dump configured with
    allow_nan=False, so the failure would arrive at the end of a long run and
    point at the writer rather than at the subset with one stratum in it.
    """
    import json
    for df in (0, -1):
        r = dp.t_and_p(0.0148, 0.004, df)
        assert r["t"] is None, df
        assert isinstance(r["p_value"], float) and r["p_value"] == 1.0, df
        assert r["degenerate"] and "degrees of freedom" in r["degenerate"], df
        json.dumps(r, allow_nan=False)


def test_t_and_p_is_the_single_source_for_all_three_scripts(dp):
    """
    ablation, benchmark and weighted must not answer this differently again.

    Checked by calling each script's own wrapper on a degenerate contrast rather
    than by reading their source: identical predictions have to come back p = 1
    from all three, which before this helper existed they did not.
    """
    import ablation
    import benchmark
    import weighted
    for mod in (ablation, benchmark, weighted):
        assert mod.dp.t_and_p is dp.t_and_p, f"{mod.__name__} has its own copy"
    assert dp.t_and_p(0.0, 0.0, 25)["p_value"] == 1.0
