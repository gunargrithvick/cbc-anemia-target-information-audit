"""
Interactive anemia screening demonstration for the research study.

This is the rewritten version of the submitted manuscript's application. Eight
things changed, and each one corresponds to a defect the audit found:

  1. FEATURES. The old header said "LEAKAGE-FREE" over [RBC, MCV, MCH], but
     MCH * RBC / 10 IS hemoglobin. The tool now uses the L6 rung from
     dataprep: RBC, MCV, RDW plus age, sex and pregnancy. No exact identity
     survives, and the demographics are things a clinician legitimately knows.
  2. LABEL. The old code cut everyone at 12.0 g/dL and used 12/10 severity
     bands. Labels now come from dataprep's WHO age-, sex- and
     pregnancy-specific thresholds.
  3. SPLIT. The old code called train_test_split twice with different
     stratify targets, so the binary and severity models were scored on
     different test sets. Both now use the single shared three-way
     dp.split3(), and both are fitted on the 60% training block only - the
     same fit rule as ablation.py and benchmark.py, so the numbers in
     models/model_info.json can be compared with the ones in the paper
     instead of being a fourth, incompatible set. The 20% validation block is
     left unused here on purpose: this script selects nothing, so touching it
     would buy a little accuracy at the cost of every cross-script
     comparison.
  4. SCALER. StandardScaler was wrapped around a Random Forest, which is
     scale-invariant, so it did nothing. Removed.
  5. HEMOGLOBIN PROMPT. The old tool asked for Hb, range-checked it, then
     never passed it to the model. It no longer asks: the point of the study
     is that Hb must not enter the inputs.
  6. REJECTION COUNTING. Physiological range checks now record every
     rejection to models/rejections.json, so the paper can report how often
     the guard fires instead of merely asserting that it exists. The guard also
     checks that age, sex and pregnancy are whole numbers, not just that they
     fall inside a range, and a damaged tally file is quarantined and named
     rather than silently replaced by a zeroed one.
  7. IMPOSSIBLE COMBINATIONS. Per-field checks cannot see them, so "male and
     pregnant" was accepted and scored. coherence_error() rejects that, and
     rejects pregnant=1 outside ages 20-44, the window in which NHANES
     ascertains pregnancy through RIDEXPRG. Ages 43 and 44 are inside the
     ascertainment window but hold no pregnant row in this cohort at all (the 76
     pregnant rows span 20-42, and only 44 of them are in the training block), so
     the window is where the ANSWER exists, not where the training evidence
     does. The rejection message says which of the two it means.
  8. INPUT RANGES. The per-field ranges were a private, narrower set than
     dataprep.integrity_screen's containment bounds, and they refused 49 real
     cohort rows - 34 of them from this forest's own training block, and 23 of
     the 49 anemic. A screening tool must not refuse the patients it exists to
     catch, and the study's own policy string says so. The three CBC prompts now
     read their limits from dp.CONTAINMENT_BOUNDS, one definition in one place.

Run:  python src/anemia_app.py
"""

import json
import sys
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import dataprep as dp
from paths import FIGURES, MODELS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BINARY_MODEL_PATH = MODELS / "anemia_binary.pkl"
SEVERITY_MODEL_PATH = MODELS / "anemia_severity.pkl"
META_PATH = MODELS / "model_info.json"
REJECT_PATH = MODELS / "rejections.json"

RUNG = "L6_identity_free_demo"
FEATURES = dp.LADDER[RUNG]          # [RBC, MCV, RDW, AGE, SEX, PREGNANT]
SEED = dp.SEED
N_FOLDS = dp.N_FOLDS       # one place, dataprep; see the note there
# 300 for both models, and it now comes from dataprep rather than from here. The
# severity forest used 400 while every other script in the project -
# ablation.py, benchmark.py, weighted.py - used 300, so the one number this file
# exists to publish could not be compared with the paper's (0.9502 at 400 against
# 0.9515 at 300). Docstring point 3 promises exactly that comparison, so the
# forest matches instead of the promise being dropped. Re-typing 300 here was the
# first fix and it left four independent literals; dp.N_TREES leaves one.
N_TREES = dp.N_TREES

# Prompt spec, keyed by the column each one fills.
#
# The three CBC ranges are dp.CONTAINMENT_BOUNDS, the same limits
# dataprep.integrity_screen reports against, not a private set. They were a
# private, narrower set (RBC 2.0-6.5, MCV 60-110, RDW 10-30) and that was a
# contradiction with the study's own stated policy: those ranges refuse 49 of the
# 12,156 cohort rows, 34 of them from the training block this forest was fitted
# on, and 23 of the 49 are anemic - a 46.9% anemia rate against the cohort's
# 9.60%. dataprep.integrity_screen's policy string argues that dropping such
# rows "would remove exactly the patients a screening tool exists to catch", and
# then the screening tool was refusing them at the prompt. The bounds are
# containment bounds now: outside them a value is a unit or transcription error,
# inside them it is a patient, and every patient in the cohort is inside except
# two. Those two are the MCVs at 35.4 fL that integrity_screen flags and keeps:
# the study's own line for "this is a unit error" is 40 fL, so the app refuses
# them at the prompt while the analysis keeps them in the fit. That asymmetry is
# deliberate and is the whole content of the report-do-not-clean policy - a
# screen that publishes a count must not also be the screen that silently
# rewrites a denominator - but it does mean the tool will refuse 2 of the 12,156
# people it was fitted on, down from 49 before this change.
#
# whole=True marks the fields where a fraction is not a value at all. The range
# test was once the only check, so "sex 1.5" and "pregnant 0.5" were accepted as
# valid codes, and an age of 4.5 fell into the gap between WHO's under-5 and 5-11
# bands, which made who_thresholds return nan while the tool printed a prediction
# anyway. NHANES records age in whole years (RIDAGEYR), so the model never saw a
# fraction.
PROMPTS = [
    (dp.RBC,  "RBC count", "10^6/uL",         *dp.CONTAINMENT_BOUNDS[dp.RBC], False),
    (dp.MCV,  "MCV",       "fL",              *dp.CONTAINMENT_BOUNDS[dp.MCV], False),
    (dp.RDW,  "RDW",       "%",               *dp.CONTAINMENT_BOUNDS[dp.RDW], False),
    (dp.AGE,  "Age",       "whole years",     1.0,  80.0, True),
    (dp.SEX,  "Sex",       "1=male 2=female", 1.0,   2.0, True),
    (dp.PREG, "Pregnant",  "0=no 1=yes",      0.0,   1.0, True),
]
# The prompts are zipped against FEATURES positionally, and the guard used to be
# len(PROMPTS) == len(FEATURES). That passes for any reordering of the ladder,
# which would have quietly filed the RBC answer in the MCV column. The column
# names are compared now.
assert [c for c, *_ in PROMPTS] == list(FEATURES), (PROMPTS, list(FEATURES))

# Cross-field coherence. Range and whole-number checks are per-field, so each of
# these combinations passed every check the tool had while being either
# impossible or outside the data the model was fitted on:
#
#   sex=1 with pregnant=1        impossible.
#   pregnant=1 at age 12 or 60   possible in the world, absent from the fit.
#
# The age window is not a clinical claim. NHANES ascertains pregnancy through
# RIDEXPRG, which is only collected for women aged 20-44, so PREGNANT=1 cannot
# occur outside it in the training data. Measured on the 12,156-row cohort:
# PREG_ASCERTAINED spans exactly ages 20-44 and is female-only, and the 76 rows
# with PREGNANT=1 span 20-42 - so 43 and 44 are inside the window NHANES asks in
# and still hold no pregnant row, and the training block holds 44 pregnant rows
# in total. The window is therefore where the QUESTION was asked, not where the
# evidence is; the reject message says so rather than claiming a training row
# exists at every accepted age. A pregnant 17-year-old is a real patient; this
# tool simply has no evidence about her and says so instead of extrapolating
# silently. dataprep.who_thresholds would hand her the pregnant 11.0 g/dL band,
# so the failure would be quiet, not loud. The window is not narrowed to 20-42,
# because 76 rows is too thin a basis for drawing an evidence boundary at all -
# widening the message is honest, narrowing the gate would be false precision.
PREG_ASCERTAINED_AGES = (20.0, 44.0)
PREG_OBSERVED_AGES = (20.0, 42.0)      # measured, not documented; see above
FEMALE = 2.0


def coherence_error(v):
    """
    Cross-field checks, on the dict of accepted answers.

    Returns (field_tag, message) for the first incoherent combination, or None.
    Kept separate from ask() so it can be reasoned about and tested without a
    terminal attached.
    """
    if v[dp.PREG] == 1.0:
        if v[dp.SEX] != FEMALE:
            return ("Pregnant:male_and_pregnant",
                    "sex = 1 (male) with pregnant = 1 is not a possible "
                    "combination")
        lo, hi = PREG_ASCERTAINED_AGES
        if not (lo <= v[dp.AGE] <= hi):
            return ("Pregnant:outside_ascertainment_window",
                    f"pregnant = 1 at age {v[dp.AGE]:g} is outside "
                    f"{lo:g}-{hi:g}, the only ages at which NHANES records "
                    f"pregnancy (RIDEXPRG), so the training data contains no "
                    f"pregnant row here and the tool will not guess")
    return None


def evidence_note(v):
    """
    A warning, not a rejection: accepted input the data is thin on.

    coherence_error() gates on the ascertainment window, which is where NHANES
    asked the question. This says where it got a positive answer. Ages 43 and 44
    pass the gate and hold no pregnant row anywhere in the cohort, so a
    prediction there is an extrapolation and is labelled one.

    Two corrections to what this used to say. It said "the training data", and
    PREG_OBSERVED_AGES is measured on the whole 12,156-row COHORT - only 44 of
    the 76 pregnant rows are in the 60% training block this forest is fitted on,
    so 20-42 overstates the fit's coverage and a pregnant 41-year-old could be
    inside the window and still absent from the fit. And 76 rows over 23 distinct
    ages is about three rows an age, so being inside the window is not evidence
    of much either; a second, unconditional caution says so rather than letting
    silence imply adequacy. Per-age training coverage is not computed here on
    purpose: it would mean loading the cohort on the interactive path to sharpen
    a warning that is already the strongest thing this function can honestly say.
    """
    if v[dp.PREG] != 1.0:
        return None
    lo, hi = PREG_OBSERVED_AGES
    if not (lo <= v[dp.AGE] <= hi):
        return (f"no pregnant patient aged {v[dp.AGE]:g} appears anywhere in the "
                f"12,156-row cohort (the 76 pregnant rows span {lo:g}-{hi:g}, "
                f"and only 44 of those are in the training block); this "
                f"prediction is an extrapolation")
    return (f"the cohort holds 76 pregnant rows across ages {lo:g}-{hi:g}, "
            f"44 of them in the training block - roughly three rows per year of "
            f"age. Being inside that span is not the same as being supported "
            f"by it; treat any pregnancy prediction as weakly evidenced")

# Derived from dataprep so the band names cannot drift apart from the labels.
SEVERITY_MAP = {i: (name if name == "Normal" else f"{name} anemia")
                for i, name in enumerate(dp.SEVERITY_NAMES_3)}


def _blank_reject_log():
    return {"attempts": 0, "accepted": 0, "rejected": 0, "by_field": {}}


def _incoherent(log):
    """Why this log cannot be trusted as a running tally, or None if it can."""
    if not isinstance(log, dict):
        return f"is a {type(log).__name__}, not an object"
    for key in ("attempts", "accepted", "rejected"):
        # `not isinstance(v, bool)` matters: bool is a subclass of int, so
        # {"attempts": true, "accepted": true, "rejected": false} passed every
        # check here - True is 1, False is 0, and 1 + 0 == 1 - and a file whose
        # counters are JSON booleans was accepted as a valid tally and then
        # incremented. The by_field loop below already excluded bools; the three
        # top-level counters did not.
        v = log.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return f"{key!r} is not a count"
    if not isinstance(log.get("by_field"), dict):
        return "'by_field' is not an object"
    # The values, not just the container. sum() over a dict holding a string
    # raised TypeError out of load_rejections, which catches only
    # JSONDecodeError and OSError - so a hand-edited "1" crashed the tool
    # instead of being quarantined, which is the one outcome this function
    # exists to prevent.
    for field, n in log["by_field"].items():
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return f"by_field[{field!r}] is not a count"
    if log["accepted"] + log["rejected"] != log["attempts"]:
        return (f"{log['accepted']} accepted + {log['rejected']} rejected "
                f"!= {log['attempts']} attempts")
    if sum(log["by_field"].values()) != log["rejected"]:
        return (f"by_field sums to {sum(log['by_field'].values())}, "
                f"not {log['rejected']}")
    return None


def load_rejections():
    """
    The running input-guard tally, or a fresh one if the file cannot be used.

    A damaged file used to be swallowed silently: the JSONDecodeError was caught,
    a zeroed log was returned, and the next accepted attempt wrote it back over
    the original. The tally the paper quotes would then have restarted at 1 with
    nothing to say it had. The bad file is moved aside and named instead, and the
    replacement carries the discontinuity in it so the reset is visible in the
    artefact rather than only in whoever happened to be watching the terminal.
    """
    if not REJECT_PATH.exists():
        return _blank_reject_log()

    try:
        with open(REJECT_PATH, encoding="utf-8") as fh:
            log = json.load(fh)
        why = _incoherent(log)
    # UnicodeDecodeError is in the list because it is neither of the other two:
    # it derives from ValueError, and json.JSONDecodeError does too but is a
    # sibling, not a parent. A rejections.json holding a stray non-UTF-8 byte -
    # the ordinary result of an interrupted write or a hand edit in the wrong
    # encoding - therefore raised straight out of the function that exists to
    # quarantine damaged files, and crashed the tool at startup.
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        log, why = None, f"could not be read ({exc})"
    if why is None:
        return log

    # Timestamped, not a fixed ".corrupt.json". A fixed name means the second
    # corruption silently destroys the evidence from the first, and Path.replace
    # overwrites without warning, so the docstring's promise that the bad file is
    # "moved aside and named" held exactly once.
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    quarantine = REJECT_PATH.with_name(f"{REJECT_PATH.stem}.corrupt.{stamp}.json")
    n = 0
    while quarantine.exists():                 # same second, twice
        n += 1
        quarantine = REJECT_PATH.with_name(
            f"{REJECT_PATH.stem}.corrupt.{stamp}-{n}.json")
    try:
        REJECT_PATH.replace(quarantine)
        moved = quarantine.name
    except OSError:
        moved = None
    print(f"\n  WARNING: {REJECT_PATH.name} {why}.")
    print(f"  {'Moved to ' + moved if moved else 'Could not move it aside'}; "
          f"the guard tally restarts from zero.")
    fresh = _blank_reject_log()
    fresh["restarted"] = {"reason": why, "previous_file": moved}
    # Written here, not left for the next accepted attempt to write. The
    # docstring above promises the reset is "visible in the artefact", and it was
    # not: the only writer is record(), so a session that opened the app, hit the
    # quarantine, looked at option 4 and quit moved the damaged file aside and
    # replaced it with nothing at all. The artefact the paper quotes was deleted
    # rather than reset, and the `restarted` block explaining why went with it.
    try:
        save_rejections(fresh)
    except OSError as exc:
        print(f"  and the replacement could not be written ({exc}); "
              f"this session's tally is in memory only.")
    return fresh


def save_rejections(log):
    """
    Write the tally, atomically.

    A plain open("w") truncates first, so an interrupt during the write leaves a
    half-written file - which is precisely the corruption load_rejections has to
    quarantine. download_data.py already writes through a .part; so does this.
    """
    tmp = REJECT_PATH.with_suffix(REJECT_PATH.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, allow_nan=False)
    tmp.replace(REJECT_PATH)


def record(log, field=None):
    """One attempt in, one outcome out. field=None means the attempt passed."""
    log["attempts"] += 1
    if field is None:
        log["accepted"] += 1
    else:
        log["rejected"] += 1
        log["by_field"][field] = log["by_field"].get(field, 0) + 1
    save_rejections(log)


def fit_one(X, y, tr, te, names=None):
    """
    Fit one Random Forest and report it honestly.

    The forest size is read from N_TREES rather than taken as an argument. It
    used to be a parameter, and both call sites passed the same module constant
    into it - a knob with one setting, which suggests to a reader that the two
    models were tuned separately when they were not. dp.N_TREES is the single
    source; anemia_app.py must not be able to disagree with the pipeline.

    X is a DataFrame on purpose: fitting with column names means sklearn will
    raise at predict time if a caller ever hands over the six values in the
    wrong order, which for a clinical tool is worth more than the microsecond
    saved by passing a bare array.

    No StandardScaler: a forest splits on thresholds, so rescaling a feature
    cannot change any split it chooses. The per-fold array is kept, not
    averaged away, because the spread is what tells a reader whether a
    difference of a point means anything.

    The cross-validation here is i.i.d. over rows and so ignores the survey
    design; it is a stability check on the training block, not an interval.
    The design-based intervals are in ablation.json and benchmark.json.
    """
    clf = RandomForestClassifier(n_estimators=N_TREES,
                                 class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    cv = cross_val_score(clf, X.iloc[tr], y[tr], scoring="accuracy",
                         cv=StratifiedKFold(N_FOLDS, shuffle=True,
                                            random_state=SEED), n_jobs=-1)
    clf.fit(X.iloc[tr], y[tr])
    pred = clf.predict(X.iloc[te])
    yte = y[te]
    maj = float(max(np.bincount(yte)) / len(yte))

    stats = {
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_sd": float(cv.std(ddof=1)),
        "cv_folds": [round(float(v), 4) for v in cv],
        "cv_note": "i.i.d. over rows, not design-based; a stability check only",
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "no_information_rate": maj,
        "lift_over_baseline": float(accuracy_score(yte, pred)) - maj,
    }
    if len(set(y)) == 2:
        proba = clf.predict_proba(X.iloc[te])[:, 1]
        stats["pr_auc"] = float(average_precision_score(yte, proba))
        stats["pr_auc_no_skill"] = float(yte.mean())
        stats["recall"] = float(recall_score(yte, pred, zero_division=0))

    print(f"  CV accuracy       {cv.mean():.4f} +/- {cv.std(ddof=1):.4f}"
          f"   folds {[round(float(v), 4) for v in cv]}")
    print(f"  test accuracy     {stats['test_accuracy']:.4f}")
    print(f"  balanced accuracy {stats['test_balanced_accuracy']:.4f}")
    print(f"  macro F1          {stats['macro_f1']:.4f}")
    print(f"  no-information    {maj:.4f}   lift "
          f"{stats['lift_over_baseline']*100:+.2f} points")
    print(classification_report(yte, pred, target_names=names,
                                zero_division=0))
    return clf, stats


def train_models():
    print("\nTraining on the shared cohort and the shared split.\n")
    df, att = dp.load_cohort()
    tr, va, te = dp.split3(df)            # ONE split, both models
    X = df[FEATURES]

    print(f"cohort {att['n_analysis']:,}   train {len(tr):,}   "
          f"val {len(va):,} (unused here)   test {len(te):,}")
    print(f"features {FEATURES}")
    print("hemoglobin is NOT among them and is never requested at predict time\n")

    print("BINARY ANEMIA (WHO age/sex/pregnancy thresholds)")
    binary_model, bin_stats = fit_one(X, df["Anemia"].to_numpy(), tr, te,
                                     names=["Normal", "Anemic"])
    joblib.dump(binary_model, BINARY_MODEL_PATH)

    print("SEVERITY (WHO bands, moderate and severe merged)")
    sev_model, sev_stats = fit_one(X, df["Severity"].to_numpy(), tr, te,
                                   names=dp.SEVERITY_NAMES_3)
    joblib.dump(sev_model, SEVERITY_MODEL_PATH)

    meta = {
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rung": RUNG,
        "features": list(FEATURES),
        "label_definition": "WHO age/sex/pregnancy-specific hemoglobin cut-offs",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_cohort": att["n_analysis"],
        "n_train": int(len(tr)),
        "n_val_unused": int(len(va)),
        "n_test": int(len(te)),
        "n_estimators": N_TREES,
        "fit_rule": "fitted on the 60% training block only, as in ablation.py "
                    "and benchmark.py",
        "shared_split": True,
        "standard_scaler": False,
        "binary": bin_stats,
        "severity": sev_stats,
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, allow_nan=False)

    print(f"saved {BINARY_MODEL_PATH.name}, {SEVERITY_MODEL_PATH.name}, "
          f"{META_PATH.name}\n")


def ask(log):
    """
    Collect one patient. Returns a DataFrame row, or None if it was rejected.

    Two layers. Per-field: numeric, in range, whole where a fraction is not a
    value. Then cross-field, in coherence_error(), because every per-field check
    can pass on a combination that cannot exist - male and pregnant was accepted
    by the old tool and handed straight to the model.

    Every rejection is recorded with the field that caused it, so the paper can
    quote a real rejection rate. Hemoglobin is deliberately not requested.

    An EOF on stdin returns None rather than raising. The module already handles
    being run non-interactively - __main__ switches matplotlib to Agg when stdout
    is not a tty - and in that mode input() raises EOFError, which used to escape
    as a traceback from inside the input guard. It is an absent answer, not a
    rejected one, so nothing is recorded against a field.
    """
    vals = {}
    for col, name, unit, lo, hi, whole in PROMPTS:
        try:
            raw = input(f"  {name} ({unit}) [{lo:g}-{hi:g}]: ").strip()
        except EOFError:
            print("\n  no input available; nothing recorded.\n")
            return None
        try:
            v = float(raw)
        except ValueError:
            record(log, f"{name}:not_a_number")
            print(f"  rejected: '{raw}' is not a number\n")
            return None
        if not (lo <= v <= hi):
            record(log, f"{name}:out_of_range")
            print(f"  rejected: {name} = {v:g} is outside {lo:g}-{hi:g}\n")
            return None
        if whole and v % 1:
            record(log, f"{name}:not_a_whole_number")
            print(f"  rejected: {name} = {v:g} must be a whole number "
                  f"({unit})\n")
            return None
        vals[col] = v

    bad = coherence_error(vals)
    if bad is not None:
        field, why = bad
        record(log, field)
        print(f"  rejected: {why}\n")
        return None

    record(log)
    return pd.DataFrame([[vals[c] for c in FEATURES]], columns=list(FEATURES))


def predict_patient():
    if not BINARY_MODEL_PATH.exists() or not SEVERITY_MODEL_PATH.exists():
        print("\nTrain the models first (option 1).\n")
        return

    binary_model = joblib.load(BINARY_MODEL_PATH)
    severity_model = joblib.load(SEVERITY_MODEL_PATH)
    log = load_rejections()

    print("\nEnter the patient. Hemoglobin is not requested - it defines the")
    print("label, so a model that received it would not be predicting anything.")
    row = ask(log)
    if row is None:
        print(f"  rejections so far: {log['rejected']} of {log['attempts']} "
              f"attempts\n")
        return

    p_anemia = float(binary_model.predict_proba(row)[0, 1])
    anemic = int(binary_model.predict(row)[0])
    sev = int(severity_model.predict(row)[0])

    # the threshold this patient would be judged against, for context only:
    # it is NOT an input, and no hemoglobin value is involved
    cut, _, _ = dp.who_thresholds([row[dp.AGE].iloc[0]], [row[dp.SEX].iloc[0]],
                                  [row[dp.PREG].iloc[0]])
    cut = float(cut[0])

    print("\n  SCREENING RESULT")
    print("  " + "-" * 52)
    print(f"  anemia            {'flagged' if anemic else 'not flagged'}"
          f"   (probability {p_anemia:.3f})")
    print(f"  severity          {SEVERITY_MAP.get(sev, f'class {sev}')}")
    if np.isnan(cut):
        # Unreachable while ask() enforces whole years, and stated rather than
        # formatted: the old code printed "nan g/dL" beside a real prediction.
        print("  WHO cut-off       no WHO band covers this age/sex/pregnancy "
              "combination")
    else:
        print(f"  WHO cut-off for this age/sex/pregnancy   {cut:.1f} g/dL")
    print("  " + "-" * 52)
    thin = evidence_note({c: float(row[c].iloc[0]) for c, *_ in PROMPTS})
    if thin is not None:
        print(f"  CAUTION: {thin}")
    print("  Screening output only. Confirm with a measured hemoglobin.")
    print(f"  input guard: {log['rejected']} rejected of {log['attempts']} "
          f"attempts\n")


def show_rejections():
    log = load_rejections()
    print("\nINPUT GUARD LOG (models/rejections.json)")
    print(f"  attempts   {log['attempts']}")
    print(f"  accepted   {log['accepted']}")
    print(f"  rejected   {log['rejected']}")
    if log["attempts"]:
        print(f"  rate       {log['rejected']/log['attempts']*100:.1f}%")
    for field, n in sorted(log["by_field"].items(), key=lambda kv: -kv[1]):
        # 40, not 28: the coherence tags are up to 37 characters
        # ("Pregnant:outside_ascertainment_window"), which broke the column.
        print(f"    {field:<40} {n}")
    if not log["attempts"]:
        print("  no attempts recorded yet")
    print()


def show_graphs():
    if not BINARY_MODEL_PATH.exists():
        print("\nTrain the models first (option 1).\n")
        return
    import matplotlib.pyplot as plt

    df, _ = dp.load_cohort()
    model = joblib.load(BINARY_MODEL_PATH)
    imp = model.feature_importances_

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # Reindexed onto the full code range, not left as whatever value_counts()
    # happened to observe. Both bar() calls pass a fixed label list against a
    # counted Series, so a class with no rows in the cohort made the two
    # arguments different lengths and matplotlib raised - and it would have
    # raised on exactly the run where the class balance was worth looking at.
    # A missing class is a zero-height bar, which is the honest picture.
    counts = df["Anemia"].value_counts().reindex([0, 1], fill_value=0)
    ax[0].bar(["Normal", "Anemic"], counts.values, color=["#4c72b0", "#c44e52"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n{v/len(df)*100:.1f}%", ha="center",
                   va="bottom", fontsize=9)
    ax[0].set_title("Binary class distribution (WHO labels)")
    ax[0].set_ylim(0, max(counts.max() * 1.18, 1.0))

    c3 = df["Severity"].value_counts().reindex(
        range(len(dp.SEVERITY_NAMES_3)), fill_value=0)
    ax[1].bar(dp.SEVERITY_NAMES_3, c3.values, color="#55a868")
    for i, v in enumerate(c3.values):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax[1].set_title(f"Severity distribution ({c3.max()/max(c3.min(),1):.0f}:1 "
                    f"imbalance)")
    ax[1].set_ylim(0, max(c3.max() * 1.18, 1.0))
    ax[1].tick_params(axis="x", labelsize=8)

    order = np.argsort(imp)
    ax[2].barh([FEATURES[i] for i in order], imp[order], color="#8172b2")
    ax[2].set_title(f"Feature importance, {RUNG}")
    ax[2].set_xlabel("mean decrease in impurity")

    plt.tight_layout()
    out = FIGURES / "app_overview.png"
    plt.savefig(out, dpi=200)
    print(f"\nsaved {out}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main():
    menu = ("\n==== ANEMIA SCREENING TOOL ====\n"
            f"features: {', '.join(FEATURES)}\n"
            "1. Train models\n"
            "2. Screen a patient\n"
            "3. Graphical analysis\n"
            "4. Input guard log\n"
            "5. Exit\n")
    actions = {"1": train_models, "2": predict_patient,
               "3": show_graphs, "4": show_rejections}
    while True:
        print(menu)
        # An EOF here is a clean exit, not a crash. The __main__ guard below
        # already expects non-interactive runs (it switches matplotlib to Agg
        # when stdout is not a tty), and in that mode the very first input()
        # raises EOFError - so the one entry point the guard was written for
        # ended in a traceback instead of the menu it was meant to drive.
        try:
            choice = input("Choice: ").strip()
        except EOFError:
            print("\nstdin closed; exiting.\n")
            break
        if choice == "5":
            print("\nDone.\n")
            break
        action = actions.get(choice)
        if action is None:
            print("\nPick 1-5.\n")
        else:
            action()


if __name__ == "__main__":
    if not sys.stdout.isatty():          # non-interactive run, e.g. a smoke test
        matplotlib.use("Agg")
    main()
