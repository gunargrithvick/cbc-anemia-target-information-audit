"""
The interactive tool's input guards, tested without a terminal attached.

Everything here is a pure function on a dict or on a log object, which is why
anemia_app.py keeps coherence_error() out of ask() in the first place. No model is
loaded and nothing is written to models/.

The guards matter more than they look. dataprep.who_thresholds would happily hand
a pregnant 17-year-old the pregnant 11.0 g/dL band, so the failure these prevent is
a quiet wrong answer, not a crash.
"""

import pytest


@pytest.fixture(scope="module")
def app():
    import anemia_app
    return anemia_app


@pytest.fixture
def ok(app, dp):
    """A coherent adult non-pregnant woman: the baseline every case perturbs."""
    return {dp.RBC: 4.5, dp.MCV: 90.0, dp.RDW: 13.0,
            dp.AGE: 30.0, dp.SEX: 2.0, dp.PREG: 0.0}


def test_the_prompts_line_up_with_the_model_columns(app, dp):
    """
    Positional zipping, so a reordered ladder must fail here, not silently.

    The old guard was len(PROMPTS) == len(FEATURES), which passes for any
    permutation - and a permutation files the RBC answer in the MCV column and
    still prints a confident prediction.
    """
    assert [c for c, *_ in app.PROMPTS] == list(dp.LADDER[app.RUNG])
    assert app.RUNG == "L6_identity_free_demo", (
        "the app no longer serves the defensible rung")


def test_a_coherent_patient_is_accepted(app, ok):
    assert app.coherence_error(ok) is None
    assert app.evidence_note(ok) is None


def test_a_pregnant_male_is_refused(app, ok, dp):
    ok[dp.SEX] = 1.0
    ok[dp.PREG] = 1.0
    field, msg = app.coherence_error(ok)
    assert field == "Pregnant:male_and_pregnant"
    assert "not a possible" in msg


@pytest.mark.parametrize("age", [12.0, 17.0, 19.0, 45.0, 60.0, 80.0])
def test_pregnancy_outside_the_ascertainment_window_is_refused(app, ok, dp, age):
    """
    NHANES asks about pregnancy only at 20-44, so PREGNANT=1 elsewhere has no fit.

    Refused rather than answered. A pregnant 17-year-old is a real patient and this
    tool has no evidence about her; the message says that instead of extrapolating.
    """
    ok[dp.AGE] = age
    ok[dp.PREG] = 1.0
    field, msg = app.coherence_error(ok)
    assert field == "Pregnant:outside_ascertainment_window"
    assert "RIDEXPRG" in msg


@pytest.mark.parametrize("age", [20.0, 30.0, 44.0])
def test_pregnancy_inside_the_window_is_accepted(app, ok, dp, age):
    ok[dp.AGE] = age
    ok[dp.PREG] = 1.0
    assert app.coherence_error(ok) is None


@pytest.mark.parametrize("age", [43.0, 44.0])
def test_the_two_ages_with_no_pregnant_training_row_are_flagged_not_refused(app, ok, dp, age):
    """
    The gap between where the question was asked and where it got a yes.

    NHANES ascertains pregnancy at 20-44; the 76 pregnant rows span 20-42. So 43
    and 44 pass the gate - narrowing it to 20-42 would be false precision on 76
    rows - and evidence_note labels the prediction an extrapolation instead.
    """
    ok[dp.AGE] = age
    ok[dp.PREG] = 1.0
    assert app.coherence_error(ok) is None
    note = app.evidence_note(ok)
    assert note and "extrapolation" in note


def test_the_two_pregnancy_windows_are_ordered_the_way_the_comment_says(app):
    a_lo, a_hi = app.PREG_ASCERTAINED_AGES
    o_lo, o_hi = app.PREG_OBSERVED_AGES
    assert a_lo <= o_lo and o_hi <= a_hi, "observed ages fall outside ascertained"
    assert (a_lo, a_hi) == (20.0, 44.0)
    assert (o_lo, o_hi) == (20.0, 42.0)


def test_whole_number_fields_are_the_ones_a_fraction_would_break(app, dp):
    """
    Age, sex and pregnant. A fraction in any of them is not a value at all.

    Age 4.5 is the one that mattered: it lands in the gap between WHO's 6-59 month
    and 5-11 year bands, so who_thresholds returned nan while the tool printed a
    prediction anyway. RBC, MCV and RDW are genuinely continuous and must stay so.
    """
    whole = {c for c, *_, w in app.PROMPTS if w}
    assert whole == {dp.AGE, dp.SEX, dp.PREG}


def test_prompt_bounds_come_from_the_shared_containment_table(app, dp):
    """
    The three lab fields must not carry a second, private set of bounds.

    dataprep.CONTAINMENT_BOUNDS is what integrity_screen reports against, so a
    divergence here would mean the app refuses values the analysis counted, or
    accepts values it flagged, with no single place that says which.
    """
    for col, _, _, lo, hi, _ in app.PROMPTS:
        if col in dp.CONTAINMENT_BOUNDS:
            assert (lo, hi) == dp.CONTAINMENT_BOUNDS[col], col


# ------------------------------------------------- the guard tally's own guard

def test_a_blank_log_is_coherent(app):
    assert app._incoherent(app._blank_reject_log()) is None


def test_a_consistent_log_is_coherent(app):
    log = {"attempts": 5, "accepted": 3, "rejected": 2,
           "by_field": {"Age:range": 1, "Pregnant:male_and_pregnant": 1}}
    assert app._incoherent(log) is None


@pytest.mark.parametrize("log,what", [
    ([], "a list instead of an object"),
    ({"attempts": "5", "accepted": 3, "rejected": 2, "by_field": {}},
     "a count that is a string"),
    ({"attempts": -1, "accepted": 0, "rejected": 0, "by_field": {}},
     "a negative count"),
    ({"attempts": 5, "accepted": 3, "rejected": 2, "by_field": []},
     "by_field is a list"),
    ({"attempts": 5, "accepted": 3, "rejected": 1, "by_field": {"Age:range": 1}},
     "accepted + rejected != attempts"),
    ({"attempts": 5, "accepted": 3, "rejected": 2, "by_field": {"Age:range": 1}},
     "by_field does not sum to rejected"),
    ({"attempts": 5, "accepted": 3, "rejected": 2, "by_field": {"Age:range": "1"}},
     "a hand-edited string inside by_field"),
    ({"attempts": 2, "accepted": 1, "rejected": 1, "by_field": {"Age:range": True}},
     "a bool where a count belongs"),
])
def test_an_incoherent_log_is_named_not_swallowed(app, log, what):
    """
    The string-in-by_field case is the one that used to crash rather than quarantine.

    load_rejections catches JSONDecodeError and OSError only, so sum() over a
    dict holding "1" raised TypeError straight out of the tool - the single
    outcome _incoherent exists to prevent.
    """
    why = app._incoherent(log)
    assert isinstance(why, str) and why, what


def test_record_and_the_tally_stay_consistent(app, tmp_path, monkeypatch):
    """
    record() twice, and the invariants _incoherent checks still hold.

    Pointed at tmp_path, so models/rejections.json is untouched - which also keeps
    the run manifest valid, since hashing that file is what used to make the
    manifest go stale the first time anybody opened the app.
    """
    monkeypatch.setattr(app, "REJECT_PATH", tmp_path / "rejections.json")
    log = app._blank_reject_log()
    app.record(log)
    app.record(log, "Pregnant:male_and_pregnant")
    assert log == {"attempts": 2, "accepted": 1, "rejected": 1,
                   "by_field": {"Pregnant:male_and_pregnant": 1}}
    assert app._incoherent(log) is None
    assert app.load_rejections() == log


def test_a_corrupt_log_is_quarantined_under_a_timestamped_name(app, tmp_path, monkeypatch):
    """
    Two corruptions must not destroy each other's evidence.

    A fixed ".corrupt.json" plus Path.replace overwrites silently, so the promise
    that the bad file is "moved aside and named" held exactly once.
    """
    target = tmp_path / "rejections.json"
    monkeypatch.setattr(app, "REJECT_PATH", target)
    for _ in range(2):
        target.write_text('{"attempts": 9, "accepted": 1, "rejected": 1, "by_field": {}}')
        fresh = app.load_rejections()
        assert fresh["attempts"] == 0
        assert fresh["restarted"]["previous_file"], "nothing was moved aside"
    quarantined = sorted(tmp_path.glob("rejections.corrupt.*.json"))
    assert len(quarantined) == 2, [p.name for p in quarantined]


def test_save_rejections_writes_through_a_part_file(app, tmp_path, monkeypatch):
    """Atomic, because a half-written tally is the corruption above."""
    target = tmp_path / "rejections.json"
    monkeypatch.setattr(app, "REJECT_PATH", target)
    app.save_rejections(app._blank_reject_log())
    assert target.exists()
    assert not list(tmp_path.glob("*.part")), "the temporary file was left behind"


def test_the_severity_names_come_from_dataprep(app, dp):
    """Band names must not drift from the labels the model was trained on."""
    assert len(app.SEVERITY_MAP) == len(dp.SEVERITY_NAMES_3)
    assert app.SEVERITY_MAP[0] == "Normal"
    for i, name in enumerate(dp.SEVERITY_NAMES_3):
        assert name in app.SEVERITY_MAP[i]


# ---------------------------------------------------------------- ask() ------
# ask() was the one untested function in this file's subject, and it is the only
# one a patient ever touches: every guard tested above is reachable only through
# it. It reads stdin, which is why it was skipped, and monkeypatching builtins
# input is the whole cost of not skipping it.
#
# The rejection tally is checked alongside the return value in every case,
# because "returns None" and "recorded one rejection against the right field"
# are two different promises and the paper quotes the second one.

@pytest.fixture
def answers(app, monkeypatch):
    """Feed ask() a script of typed answers; returns the log it wrote to."""
    def feed(seq):
        it = iter(seq)

        def fake_input(_prompt=""):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        monkeypatch.setattr("builtins.input", fake_input)
        return app._blank_reject_log()
    return feed


def test_ask_accepts_a_coherent_patient_in_the_column_order_the_model_expects(
        app, dp, answers):
    log = answers(["4.5", "90", "13", "30", "2", "0"])
    row = app.ask(log)
    assert row is not None
    assert list(row.columns) == list(app.FEATURES)
    assert row.iloc[0].tolist() == [4.5, 90.0, 13.0, 30.0, 2.0, 0.0]
    assert (log["attempts"], log["accepted"], log["rejected"]) == (1, 1, 0)
    assert log["by_field"] == {}


def test_ask_on_a_closed_stdin_records_nothing_at_all(app, answers):
    """
    An absent answer is not a rejected one, and it used to be a traceback.

    The __main__ guard already expects non-interactive runs - it switches
    matplotlib to Agg when stdout is not a tty - and in exactly that mode the
    first input() raised EOFError and escaped through the numeric guard. Nothing
    may be recorded: an EOF inflating the rejection tally would corrupt the
    rejection rate the paper reports, and it would do it on automated runs only,
    which is where nobody is watching the terminal.
    """
    log = answers([])
    assert app.ask(log) is None
    assert (log["attempts"], log["accepted"], log["rejected"]) == (0, 0, 0)
    assert log["by_field"] == {}


def test_ask_treats_an_eof_partway_through_the_same_way(app, answers):
    """Three answers then the pipe closes: still an absent attempt, not a bad one."""
    log = answers(["4.5", "90", "13"])
    assert app.ask(log) is None
    assert log["attempts"] == 0, log


@pytest.mark.parametrize("script,field", [
    (["nonsense", "90", "13", "30", "2", "0"], "RBC count:not_a_number"),
    (["", "90", "13", "30", "2", "0"], "RBC count:not_a_number"),
    (["99", "90", "13", "30", "2", "0"], "RBC count:out_of_range"),
    (["4.5", "5", "13", "30", "2", "0"], "MCV:out_of_range"),
    (["4.5", "90", "13", "30.5", "2", "0"], "Age:not_a_whole_number"),
    (["4.5", "90", "13", "30", "1.5", "0"], "Sex:not_a_whole_number"),
    (["4.5", "90", "13", "30", "3", "0"], "Sex:out_of_range"),
    (["4.5", "90", "13", "30", "1", "1"], "Pregnant:male_and_pregnant"),
    (["4.5", "90", "13", "17", "2", "1"],
     "Pregnant:outside_ascertainment_window"),
])
def test_ask_rejects_and_files_the_rejection_under_the_field_that_caused_it(
        app, answers, script, field):
    """
    The field tag is the assertion, not just the None.

    models/rejections.json is keyed by these strings and the README promises a
    rejection rate broken down by cause, so a rejection filed under the wrong
    field is a wrong number in the paper rather than a wrong branch in the code.
    An empty line is included because "" is falsy and float("") raises the same
    ValueError as "nonsense" - a guard written as `if not raw` would have skipped
    the tally.
    """
    log = answers(script)
    assert app.ask(log) is None
    assert (log["attempts"], log["accepted"], log["rejected"]) == (1, 0, 1)
    assert log["by_field"] == {field: 1}


def test_ask_stops_at_the_first_bad_field_and_does_not_double_count(app, answers):
    """Two bad answers, one rejection: the loop returns, it does not collect."""
    log = answers(["99", "5", "13", "30", "2", "0"])
    assert app.ask(log) is None
    assert log["rejected"] == 1
    assert sum(log["by_field"].values()) == 1
    assert "RBC count:out_of_range" in log["by_field"]


def test_ask_accepts_the_boundary_values_the_prompts_advertise(app, answers):
    """
    The printed range is inclusive, so the endpoints have to be accepted.

    A `lo < v < hi` guard would reject the exact numbers the prompt tells the
    user are allowed, and dp.CONTAINMENT_BOUNDS is the same interval
    dataprep.integrity_screen reports against - so the app and the screen would
    disagree about the same value.
    """
    lo = [str(p[3]) for p in app.PROMPTS[:3]] + ["20", "2", "1"]
    hi = [str(p[4]) for p in app.PROMPTS[:3]] + ["44", "2", "1"]
    for script in (lo, hi):
        log = answers(script)
        assert app.ask(log) is not None, script
        assert log["rejected"] == 0, script
