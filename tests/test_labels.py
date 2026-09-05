"""
The label. Every number the paper reports is a function of these thresholds, so
they are worth pinning to the WHO table itself rather than to what the code
currently returns.

WHO Hb cut-offs for anemia (g/dL), the 2011 haemoglobin concentrations document:

  6-59 months      11.0     moderate 10.0   severe 7.0
  children 5-11    11.5     moderate 11.0   severe 8.0
  children 12-14   12.0     moderate 11.0   severe 8.0
  women 15+        12.0     moderate 11.0   severe 8.0
  pregnant women   11.0     moderate 10.0   severe 7.0
  men 15+          13.0     moderate 11.0   severe 8.0

There is no band under 6 months, so a row recorded as age 0 gets nan, not the
nearest band.
"""

import numpy as np
import pytest


def cuts(dp, age, sex, preg):
    """who_thresholds on one row, as three floats."""
    arr = lambda x: np.array([x], dtype=float)
    return tuple(float(v[0]) for v in
                 dp.who_thresholds(arr(age), arr(sex), arr(preg)))


BANDS = [
    # age, sex, pregnant, (normal, moderate, severe), what it is
    (1, 1, 0, (11.0, 10.0, 7.0), "6-59 months, male"),
    (4, 2, 0, (11.0, 10.0, 7.0), "6-59 months, female"),
    (5, 1, 0, (11.5, 11.0, 8.0), "children 5-11, lower edge"),
    (11, 2, 0, (11.5, 11.0, 8.0), "children 5-11, upper edge"),
    (12, 1, 0, (12.0, 11.0, 8.0), "children 12-14, lower edge"),
    (14, 2, 0, (12.0, 11.0, 8.0), "children 12-14, upper edge"),
    (15, 1, 0, (13.0, 11.0, 8.0), "men 15+, lower edge"),
    (80, 1, 0, (13.0, 11.0, 8.0), "men, oldest"),
    (15, 2, 0, (12.0, 11.0, 8.0), "women 15+, lower edge"),
    (80, 2, 0, (12.0, 11.0, 8.0), "women, oldest"),
    (30, 2, 1, (11.0, 10.0, 7.0), "pregnant"),
]


@pytest.mark.parametrize("age,sex,preg,want,what", BANDS)
def test_bands_match_the_who_table(dp, age, sex, preg, want, what):
    assert cuts(dp, age, sex, preg) == want, what


def test_the_bands_are_not_sex_blind_for_adults(dp):
    """The submitted paper's flat Hb<12 is this test's opposite."""
    assert cuts(dp, 30, 1, 0)[0] == 13.0
    assert cuts(dp, 30, 2, 0)[0] == 12.0


def test_pregnancy_overrides_only_adult_women(dp):
    """
    PREGNANT=1 lowers the cut for women 15+ and is ignored everywhere else.

    Ignored, not an error: NHANES ascertains RIDEXPRG only for women 20-44, so the
    flag is 0 for everyone else by construction and a 1 outside that range comes
    from hand-entered input. anemia_app.coherence_error rejects those before they
    reach here (see test_app_guards), which is why this function can afford to
    treat the impossible combination as simply not triggering the override rather
    than raising inside a vectorised call.
    """
    assert cuts(dp, 30, 2, 1)[0] == 11.0
    assert cuts(dp, 30, 1, 1)[0] == 13.0, "a pregnant male got the pregnant cut"
    assert cuts(dp, 12, 2, 1)[0] == 12.0, "a pregnant 12-year-old got the adult pregnant cut"


NAN_CASES = [
    (0, 1, 0, "age 0: under 6 months, and WHO publishes no band"),
    (0.3, 1, 0, "3 months old"),
    (4.5, 1, 0, "gap between 6-59 months and children 5-11"),
    (11.5, 1, 0, "gap between children 5-11 and children 12-14"),
    (14.5, 1, 0, "gap between children 12-14 and adult"),
    (30, 3, 0, "adult with a sex code that is neither 1 nor 2"),
    (np.nan, 1, 0, "missing age"),
    (30, np.nan, 0, "missing sex"),
]


@pytest.mark.parametrize("age,sex,preg,what", NAN_CASES)
def test_uncovered_rows_get_nan_not_the_nearest_band(dp, age, sex, preg, what):
    """
    These are the four kinds of input who_thresholds' docstring says return nan.

    Folding them into the nearest band would be the silent failure: a 3-month-old
    scored against the 6-59 month cut, or an adult of unknown sex scored as a
    woman. nan is the honest answer and _label drops the row.
    """
    assert all(np.isnan(c) for c in cuts(dp, age, sex, preg)), what


@pytest.mark.slow
def test_no_analysis_row_survived_with_a_nan_threshold(cohort):
    """_label's dropna, checked on the delivered cohort rather than trusted."""
    df = cohort[0]
    for col in ("WHO_NORMAL", "WHO_MODERATE", "WHO_SEVERE"):
        assert df[col].notna().all(), f"{col} has nan in the analysis cohort"


@pytest.mark.slow
def test_severity_is_nested_inside_the_binary_label(cohort):
    """Severity > 0 exactly when Anemia = 1, or the two heads disagree."""
    df = cohort[0]
    assert ((df["Severity"] > 0).astype(int) == df["Anemia"]).all()
    assert ((df["Severity4"] > 0).astype(int) == df["Anemia"]).all()
    # Severity is Severity4 with Moderate and Severe merged, because Severe is 26
    # rows and will not support a fold.
    assert set(df["Severity"].unique()) <= {0, 1, 2}
    assert set(df["Severity4"].unique()) <= {0, 1, 2, 3}
    merged = df["Severity4"].clip(upper=2)
    assert (merged == df["Severity"]).all(), "Severity is not Severity4 with 2,3 merged"


@pytest.mark.slow
def test_the_flat_twelve_cut_is_wrong_for_more_than_half_the_cohort(cohort):
    """
    The submitted paper's single Hb<12 threshold, measured against WHO.

    Two different numbers, and the paper needs both. 6,895 rows - 56.7% - are
    people for whom 12.0 is not the correct cut-off at all: adult men (13.0),
    under-12s (11.0 or 11.5), pregnant women (11.0). Of those, 540 rows actually
    change label, because most people sit far from either threshold. So the flat
    cut is wrong in principle for the majority and wrong in outcome for 4.4%, and
    quoting only the second understates the design error while quoting only the
    first overstates its effect.
    """
    df = cohort[0]
    wrong_cut = int((df["WHO_NORMAL"].to_numpy() != 12.0).sum())
    changed = int((df["Anemia"].to_numpy() != df["Anemia_hb12"].to_numpy()).sum())
    assert wrong_cut == 6895, wrong_cut
    assert changed == 540, changed
    assert wrong_cut / len(df) > 0.5
