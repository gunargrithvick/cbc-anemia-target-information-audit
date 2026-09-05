"""
The leakage ladder: which rungs can rebuild hemoglobin and which cannot.

dataprep.LADDER is the study's design. These tests re-derive the leakage status of
every rung from dp.HGB_ROUTES instead of trusting the rung names, because the names
are claims - "L4_identity_free" is an assertion about arithmetic, and this is where
it gets checked.

The interesting assertion is the negative one. L4, L5 and L6 cannot rebuild
hemoglobin, and they still rebuild HEMATOCRIT exactly, because they keep MCV and
RBC. That is stated here as a property of the design rather than hidden, because
the paper's claim is "no route to the label", not "no route to anything".
"""

import pytest


def tag(dp, cols):
    """The leakage class of a feature set, derived, not looked up."""
    have = set(cols)
    if dp.HGB in have and set(dp.DEMO_FEATURES) <= have:
        return "EXACT"
    if dp.HGB in have:
        return "LABEL"
    if any(route <= have for route in dp.HGB_ROUTES):
        return "IDENT"
    if {dp.MCV, dp.RBC} <= have:
        return "hct"
    return "-"


EXPECTED = {
    "L0_hgb_direct":         "LABEL",
    "L1_paper_leaky":        "IDENT",
    "L1_paper_leaky_demo":   "IDENT",
    "L2_mchc_hct":           "IDENT",
    "L3_hct_only":           "hct",
    "L4_identity_free":      "hct",
    "L5_identity_free_plus": "hct",
    "L6_identity_free_demo": "hct",
    "L7_hgb_demo":           "EXACT",
}


def test_the_ladder_is_the_nine_rungs_the_paper_describes(dp):
    assert list(dp.LADDER) == list(EXPECTED)


@pytest.mark.parametrize("rung", list(EXPECTED))
def test_each_rung_has_the_leakage_class_its_name_claims(dp, rung):
    assert tag(dp, dp.LADDER[rung]) == EXPECTED[rung]


def test_hemoglobin_appears_only_where_it_is_meant_to(dp):
    """L0 and L7 are the leaky ceilings. Anywhere else is a bug, not a rung."""
    with_hgb = {r for r, c in dp.LADDER.items() if dp.HGB in c}
    assert with_hgb == {"L0_hgb_direct", "L7_hgb_demo"}


def test_the_identity_free_rungs_really_have_no_route_to_hemoglobin(dp):
    """
    Checked against all four routes and against every subset of the rung.

    Not just "MCH and MCHC are absent" - the test enumerates the rung's subsets
    and requires none of them to be a superset of any route, so adding a feature
    to L5 that happens to complete a route fails here rather than in a reviewer's
    email.
    """
    from itertools import combinations
    for rung in ("L4_identity_free", "L5_identity_free_plus", "L6_identity_free_demo"):
        cols = set(dp.LADDER[rung])
        assert dp.HGB not in cols
        for k in range(2, len(cols) + 1):
            for sub in combinations(sorted(cols), k):
                for route in dp.HGB_ROUTES:
                    assert not route <= set(sub), f"{rung} subset {sub} completes {set(route)}"


def test_the_identity_free_rungs_still_rebuild_hematocrit(dp):
    """
    The limitation the paper has to state, asserted so it stays stated.

    MCV*RBC/10 is HCT to a rounding error, and L4-L6 all keep MCV and RBC. So
    "identity-free" means free of a route to the LABEL, not free of every analyser
    identity. If a future rung dropped MCV or RBC this test would fail and the
    claim in the paper could be strengthened - which is why the assertion is here
    and not a comment.
    """
    for rung in ("L3_hct_only", "L4_identity_free", "L5_identity_free_plus",
                 "L6_identity_free_demo"):
        assert {dp.MCV, dp.RBC} <= set(dp.LADDER[rung]), rung


def test_every_rung_uses_only_columns_the_cohort_has(dp):
    known = set(dp.ALL_FEATURE_COLS) | set(dp.DEMO_FEATURES)
    for rung, cols in dp.LADDER.items():
        assert set(cols) <= known, f"{rung} names {set(cols) - known}"
        assert len(cols) == len(set(cols)), f"{rung} lists a column twice"


def test_the_like_for_like_pair_differs_by_exactly_one_feature(dp):
    """
    L1 vs L4 is the paper's headline contrast, so the swap has to be clean.

    Same feature count, one column exchanged: MCH out, RDW in. If the two rungs
    differed in size the contrast would confound leakage with model capacity, and
    the reviewer's first question would be which one moved the number.
    """
    a, b = set(dp.LADDER["L1_paper_leaky"]), set(dp.LADDER["L4_identity_free"])
    assert len(a) == len(b) == 3
    assert a - b == {dp.MCH}
    assert b - a == {dp.RDW}


@pytest.mark.slow
def test_the_results_ladder_is_the_source_ladder(ablation_json, dp):
    """
    The desync guard for this file: ablation.json's rungs against dp.LADDER.

    A run older than the source is exactly the state this project was found in, and
    a renamed or added rung is the version of it that a reader cannot see, because
    both the JSON and the code look internally consistent.
    """
    assert list(ablation_json["rungs"]) == list(dp.LADDER)
    for rung, block in ablation_json["rungs"].items():
        assert list(block["features"]) == list(dp.LADDER[rung]), rung
