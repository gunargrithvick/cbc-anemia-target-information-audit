"""Fast unit checks for small helpers used by the analysis stages.

The contract tests exercise the delivered artefacts and the existing tests cover
the core data invariants.  These checks fill the gap between those two layers:
they call the deterministic statistical, thresholding, survey, and estimator
helpers directly on tiny arrays, so a local refactor can fail quickly without
waiting for the full NHANES reproduction.
"""

import numpy as np
import pandas as pd
import pytest


def test_dataprep_small_design_helpers(dp):
    strata = np.array([1, 1, 2, 2, 3])
    psu = np.array([1, 2, 1, 2, 1])
    assert dp.design_df(strata, psu) == 2

    sampled = dp.psu_bootstrap(strata, psu, np.random.default_rng(42))
    assert sampled.shape == (len(strata),)
    assert sampled.min() >= 0
    assert sampled.max() < len(strata)

    base = np.array([10., 20., 30., 40., 50.])
    replicate = dp.rao_wu_weights(strata, psu, np.random.default_rng(42), base)
    assert replicate.shape == base.shape
    assert np.isfinite(replicate).all()
    # The third stratum has one PSU, so the implementation carries that PSU's
    # base weight through unchanged rather than inventing a variance replicate.
    assert replicate[-1] == base[-1]

    assert dp.holm([0.01, 0.04, 0.20]) == pytest.approx([0.03, 0.08, 0.20])


def test_benchmark_model_declarations_do_not_scale_trees():
    import benchmark
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    models = benchmark.models()
    assert set(models) == {
        "majority", "stratified", "logreg", "svm_rbf", "dtree", "rf",
        "extratrees", "histgb", "xgboost",
    }
    assert isinstance(models["rf"], RandomForestClassifier)
    assert isinstance(models["logreg"], Pipeline)
    assert "standardscaler" not in repr(models["rf"]).lower()


def test_benchmark_threshold_and_calibration_helpers_cover_regular_and_constant_scores():
    import benchmark

    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.20, 0.65, 0.90, 0.35, 0.75])
    thresholds = benchmark.pick_thresholds(y, p)
    assert set(thresholds) == {"default", "f1", "youden", "spec90"}
    assert 0.0 <= thresholds["f1"]["threshold"] <= 1.0
    assert np.isfinite(benchmark.calibration(y, p)["brier"])

    constant = benchmark.pick_thresholds(y, np.zeros(len(y)))
    assert constant["f1"]["grid_degenerate"] is True
    assert constant["f1"]["threshold"] == 0.5
    assert "unavailable" in constant["youden"]


def test_benchmark_resampling_and_mcnemar_helpers_are_deterministic():
    import benchmark

    a = benchmark.row_indices(8, n=12, seed=7)
    b = benchmark.row_indices(8, n=12, seed=7)
    assert np.array_equal(a, b)
    assert a.shape == (12, 8)
    assert benchmark.row_se(
        lambda i: 0.5 if i is None else float(np.mean(i)), 8, n=12) > 0

    y = np.array([0, 1, 0, 1, 0, 1])
    pa = np.array([0, 0, 0, 1, 1, 1])
    pb = np.array([0, 1, 1, 1, 0, 0])
    result = benchmark.mcnemar_test(y, pa, pb)
    assert result["a_right_b_wrong"] + result["a_wrong_b_right"] > 0
    assert np.isfinite(result["p_value"])


def test_ablation_paired_helpers_use_after_minus_before_sign():
    import ablation

    y = np.array([0, 1, 0, 1])
    before = np.array([0, 0, 0, 0])
    after = np.array([0, 1, 1, 0])
    assert ablation._acc_diff(y, before, after) == pytest.approx(0.0)

    result = ablation.paired_row_test(y, before, after, n=32, seed=3)
    assert set(result) == {"se", "ci95", "p_value"}
    assert len(result["ci95"]) == 2
    assert 0.0 <= result["p_value"] <= 1.0

    identical = ablation.paired_row_test(y, before, before, n=32, seed=3)
    assert identical["p_value"] == 1.0


def test_weighted_helpers_handle_normal_and_empty_domains():
    import weighted

    assert weighted.weight_cv([1., 2., 3.]) == pytest.approx(
        np.std([1., 2., 3.]) / 2.0)
    assert weighted.weight_cv([]) is None

    y = np.array([0., 1., 0., 1.])
    w = np.array([1., 2., 3., 4.])
    strata = np.array([1, 1, 2, 2])
    psu = np.array([1, 2, 1, 2])
    result = weighted.svy_prop(y, w, strata, psu)
    assert result["n_obs"] == 4
    assert 0.0 < result["prop_weighted"] < 1.0
    assert np.isfinite(result["se"])

    empty = weighted.svy_prop(y, w, strata, psu, domain=np.zeros(4))
    assert empty["undefined"] is True
    assert empty["prop_weighted"] is None

    frame = pd.DataFrame({"RIDAGEYR": [2, 16, 55, 72],
                          "RIAGENDR": [1, 2, 2, 1],
                          "PREGNANT": [0, 0, 0, 0]})
    domains = weighted.domains(frame)
    assert set(domains) == {
        "all", "men 15+", "women 15+ non-pregnant", "pregnant",
        "children 1-4", "children 5-11", "children 12-14", "age 15-49",
        "age 50-69", "age 70+",
    }
    assert int(domains["all"].sum()) == len(frame)
