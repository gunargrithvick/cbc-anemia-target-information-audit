"""
The download integrity check, tested against the failure it was written for.

download_data.problem() used to be three tests - magic bytes, a minimum size, and
a multiple of 80 - and the third one is why this file exists. A dropped connection
truncates at an arbitrary byte, so roughly 1 cut in 80 lands on a record boundary
and passed all three, including the exactly-half-file cut the module's own
docstring uses as its example. What the check does now is parse the transport
file's declared geometry and compare the delivered byte count to it, so these
tests build real truncations from a real file and require every one of them to be
refused.

Nothing here writes inside data/. The real files are read and the mutilated copies
go to pytest's tmp_path.
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dd():
    import download_data
    return download_data


@pytest.fixture(scope="module")
def pristine(dd):
    """The bytes of one real file, read once, plus its parsed geometry."""
    p = dd.DATA / "P_CBC.XPT"
    if not p.exists():
        pytest.skip("P_CBC.XPT not downloaded; run python src/download_data.py")
    info, why = dd.layout(p)
    assert why is None, why
    return p.read_bytes(), info


def test_every_real_file_is_accepted(dd):
    """
    The check has to pass the eight files the study actually uses.

    A truncation detector that rejects good data is worse than none, because the
    next person's fix is to delete the check.
    """
    missing = [n for n in dd.EXPECT if not (dd.DATA / n).exists()]
    if missing:
        pytest.skip(f"not downloaded: {missing}")
    for name in dd.EXPECT:
        assert dd.problem(dd.DATA / name) is None, f"{name}: {dd.problem(dd.DATA / name)}"


def test_the_parsed_geometry_matches_the_expect_table(dd):
    """
    layout() derives observation count from the namestr records, not from EXPECT.

    Independent corroboration, not circular: the four DEMO files' observation
    counts here are 15,560 / 9,756 / 10,175 / 9,971, which are the published
    NHANES interviewed-sample sizes for those cycles. A parser that agreed with
    EXPECT but not with NHANES would be wrong in a way EXPECT alone cannot show.
    """
    for name, (size, obs, nvars) in dd.EXPECT.items():
        p = dd.DATA / name
        if not p.exists():
            continue
        info, why = dd.layout(p)
        assert why is None, f"{name}: {why}"
        assert p.stat().st_size == size, name
        assert info["observations"] == obs, name
        assert info["variables"] == nvars, name


def test_every_record_aligned_truncation_is_refused(dd, pristine, tmp_path):
    """
    200 cuts, each on an 80-byte boundary, every one rejected.

    These are exactly the cuts the old three-test check accepted. The observation
    length of P_CBC.XPT is 176 bytes, so lcm(80, 176) = 880: one aligned cut in
    eleven also leaves a whole number of observations and cannot be caught
    structurally at all. Those are the ones the EXPECT table catches, which is why
    the copy keeps its real name here.
    """
    raw, _ = pristine
    n = len(raw)
    p = tmp_path / "P_CBC.XPT"
    for k in range(1, 201):
        p.write_bytes(raw[:n - 80 * k])
        assert dd.problem(p) is not None, f"accepted a file {80 * k} bytes short"


def test_the_half_file_from_the_docstring_is_refused(dd, pristine, tmp_path):
    """The motivating example, named because a fix that misses it is not a fix."""
    raw, _ = pristine
    p = tmp_path / "P_CBC.XPT"
    p.write_bytes(raw[:len(raw) // 2])
    why = dd.problem(p)
    assert why is not None
    assert "2,427,760" in why or "mid-record" in why, why


def test_structure_alone_catches_most_cuts_in_an_unknown_file(dd, pristine, tmp_path):
    """
    The honest measurement of what the geometry tests do WITHOUT the EXPECT table.

    Under a filename EXPECT does not know, 182 of the same 200 aligned truncations
    are still refused - by the whole-number-of-observations test and by the
    blank-padding test. This pins 182 exactly. It used to assert `>= 180` under a
    docstring that said 182, which is a two-cut drift the test was written to
    notice and could not: a check that weakens by two cases reads as a pass, and
    so does one that strengthens to 200, and those are different events.

    The 18 that get through are not a scatter. They are exactly the cuts with
    k = 5 (mod 11): the observation length of P_CBC.XPT is 176 bytes and
    lcm(80, 176) = 880, so one aligned cut in eleven also leaves a whole number of
    observations, and the offset within that class is fixed by the file's own
    length. The residue is asserted too, because "18 got through" and "18 got
    through for the reason the docstring gives" are different claims, and only the
    second one licenses calling this a known 9% blind spot. A ninth file added to
    CYCLES without an EXPECT entry inherits that blind spot at whatever residue
    its own observation length produces.
    """
    raw, _ = pristine
    n = len(raw)
    p = tmp_path / "UNKNOWN.XPT"
    missed = []
    for k in range(1, 201):
        p.write_bytes(raw[:n - 80 * k])
        if dd.problem(p) is None:
            missed.append(k)
    assert 200 - len(missed) == 182, (
        f"{200 - len(missed)}/200 caught without an EXPECT entry, not 182; "
        f"the cuts that got through were {missed}")
    assert {k % 11 for k in missed} == {5}, missed
    # And the named file, which does have an EXPECT entry, catches all 200. That
    # is the whole value of the table, stated next to the cost of not being in it.
    named = tmp_path / "P_CBC.XPT"
    for k in missed:
        named.write_bytes(raw[:n - 80 * k])
        assert dd.problem(named) is not None, f"EXPECT missed a {80 * k}-byte cut"


@pytest.mark.parametrize("cut,what", [
    (137, "an unaligned cut mid-record"),
    (176, "exactly one observation short"),
    (1, "one byte short"),
])
def test_unaligned_truncations_are_refused(dd, pristine, tmp_path, cut, what):
    raw, _ = pristine
    p = tmp_path / "UNKNOWN.XPT"
    p.write_bytes(raw[:len(raw) - cut])
    assert dd.problem(p) is not None, what


@pytest.mark.parametrize("keep,what", [
    (0, "an empty file"),
    (80, "one record"),
    (400, "five records: less than the header block"),
])
def test_files_cut_inside_the_header_are_refused(dd, pristine, tmp_path, keep, what):
    raw, _ = pristine
    p = tmp_path / "UNKNOWN.XPT"
    p.write_bytes(raw[:keep])
    assert dd.problem(p) is not None, what


def test_an_html_error_page_is_refused(dd, tmp_path):
    """CDC answers a bad URL with a page, and a page is not a transport file."""
    p = tmp_path / "P_CBC.XPT"
    p.write_bytes(b"<!DOCTYPE html><html><head><title>404</title></head></html>" * 1000)
    assert dd.problem(p) is not None


def test_a_flipped_byte_is_not_caught_and_that_is_the_documented_limit(dd, pristine, tmp_path):
    """
    The non-claim, asserted so it cannot quietly become a claim.

    Corrupting a value in place leaves the size, the geometry and the observation
    count untouched, so none of the six tests can see it. Catching that needs
    published checksums of the CDC files, which this project does not have.
    problem()'s docstring says so; this test is what keeps the docstring true. If
    a future change does add checksums, this test fails and should be deleted.
    """
    raw, _ = pristine
    b = bytearray(raw)
    b[len(raw) // 2] ^= 0xFF
    p = tmp_path / "P_CBC.XPT"
    p.write_bytes(bytes(b))
    assert dd.problem(p) is None, (
        "problem() now detects in-place corruption; update its docstring, which "
        "states it cannot, and delete this test")


def test_every_cycle_declares_the_files_expect_knows(dd):
    """CYCLES and EXPECT must not drift apart, or a new cycle gets no size check."""
    named = set()
    for cycle in dd.CYCLES:
        named |= {p.name for p in map(lambda f: dd.DATA / f, dd.cycle_files(cycle))}
    assert named == set(dd.EXPECT), (
        f"in CYCLES but not EXPECT: {sorted(named - set(dd.EXPECT))}; "
        f"in EXPECT but not CYCLES: {sorted(set(dd.EXPECT) - named)}")
