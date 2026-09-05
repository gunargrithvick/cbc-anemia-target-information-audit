"""
The analyser identities, which are the whole premise of the study.

If these fail, the paper's claim is wrong. They are arithmetic, not statistics,
so the tolerances are rounding tolerances: NHANES publishes MCV to 0.1 fL, MCH to
0.1 pg, MCHC to 0.1 g/dL, so a reconstruction can be off by what those roundings
propagate to and no more.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_three_identities_hold_to_rounding(cohort, dp):
    """
    MCV = HCT/RBC*10, MCH = HGB/RBC*10, MCHC = HGB/HCT*100.

    Not against a guessed tolerance. NHANES prints HGB, HCT, MCV, MCH and MCHC to
    0.1 and RBC to 0.01, so each row has its OWN bound: half a step of slack for
    the printed quotient, plus what half a step on each input propagates to
    through the quotient's derivatives. For MCV = 10*HCT/RBC that is

        0.05  +  10*0.05/RBC  +  MCV*0.005/RBC

    and the same shape for the other two. Median bound is 0.25 fL / 0.19 pg /
    0.21 g/dL, median residual is 0.068 / 0.058 / 0.067 - a quarter of the room
    rounding alone allows. Between 0.6% and 0.9% of rows fall outside their own
    bound, index by index (MCV 0.63%, MCH 0.85%, MCHC 0.93%); those are the
    internally inconsistent records, reported in identity_audit.json and not
    repaired, and this test asserts their share stays under 2% rather than
    pretending they do not exist. The range is given per index because a single
    "about 0.7%" was the MCV figure standing in for all three, and MCHC is half
    again as bad.
    """
    df = cohort[0]
    hgb, rbc, hct = (df[c].to_numpy() for c in (dp.HGB, dp.RBC, dp.HCT))
    mcv, mch, mchc = (df[c].to_numpy() for c in (dp.MCV, dp.MCH, dp.MCHC))
    half = 0.05        # half a printed step, for everything published to 0.1
    half_rbc = 0.005   # RBC is published to 0.01
    cases = [
        ("MCV", mcv, hct / rbc * 10,
         half + 10 * half / rbc + mcv * half_rbc / rbc),
        ("MCH", mch, hgb / rbc * 10,
         half + 10 * half / rbc + mch * half_rbc / rbc),
        ("MCHC", mchc, hgb / hct * 100,
         half + 100 * half / hct + mchc * half / hct),
    ]
    for name, printed, rebuilt, bound in cases:
        err = np.abs(printed - rebuilt)
        inside = float((err <= bound).mean())
        assert inside > 0.98, f"{name}: only {inside:.4f} of rows explained by rounding"
        assert np.median(err) < np.median(bound), (
            f"{name}: median residual {np.median(err):.4f} exceeds the median "
            f"rounding bound {np.median(bound):.4f}")


def test_every_declared_route_recovers_hemoglobin(cohort, dp):
    """dp.HGB_ROUTES is the authoritative list; each member must actually work."""
    df = cohort[0]
    hgb = df[dp.HGB].to_numpy()
    indices = [dp.HGB, dp.RBC, dp.HCT, dp.MCV, dp.MCH, dp.MCHC]
    assert len(dp.HGB_ROUTES) == 4, dp.HGB_ROUTES
    for cols in dp.HGB_ROUTES:
        assert dp.HGB not in cols, f"{cols} contains hemoglobin itself"
        assert cols <= set(indices), f"{cols} is not inside the red-cell block"
    # The four closed forms, spelled out here independently of the source so this
    # is a check and not a restatement.
    v = {c: df[c].to_numpy() for c in indices}
    routes = {
        frozenset({dp.MCH, dp.RBC}): v[dp.MCH] * v[dp.RBC] / 10,
        frozenset({dp.MCHC, dp.HCT}): v[dp.MCHC] * v[dp.HCT] / 100,
        frozenset({dp.MCHC, dp.MCV, dp.RBC}): v[dp.MCHC] * v[dp.MCV] * v[dp.RBC] / 1000,
        frozenset({dp.MCH, dp.HCT, dp.MCV}): v[dp.MCH] * v[dp.HCT] / v[dp.MCV],
    }
    assert {frozenset(r) for r in dp.HGB_ROUTES} == set(routes), (
        "HGB_ROUTES and this test disagree about which routes exist")
    for cols, rebuilt in routes.items():
        mae = float(np.mean(np.abs(hgb - rebuilt)))
        assert mae < 0.05, f"{set(cols)} gives MAE {mae:.4f} g/dL"


def test_routes_reproduce_the_who_label(cohort, dp):
    """A route is only leakage if the LABEL comes back, not just the number."""
    df = cohort[0]
    cut = df["WHO_NORMAL"].to_numpy()
    truth = df["Anemia"].to_numpy()
    rebuilt = df[dp.MCH].to_numpy() * df[dp.RBC].to_numpy() / 10
    agree = float(((rebuilt < cut).astype(int) == truth).mean())
    assert agree > 0.99, f"MCH*RBC/10 reproduces the label on only {agree:.4f}"


def test_hematocrit_is_recoverable_too(cohort, dp):
    """MCV*RBC/10 rebuilds HCT, which is how a feature list looks clean and is not."""
    df = cohort[0]
    mae = float(np.mean(np.abs(df[dp.HCT].to_numpy()
                              - df[dp.MCV].to_numpy() * df[dp.RBC].to_numpy() / 10)))
    assert mae < 0.05, f"MAE {mae:.4f}"
