"""
Step 8 - one command that reproduces every number and every figure.

The scripts are deliberately runnable on their own, but a reader who wants to
check the paper should not have to know the order. This runs them in dependency
order, times each stage, verifies the artefacts it was supposed to write, and
records a manifest with library versions and a SHA-256 of every input, every
output, and the source that turned one into the other.

The manifest is the point. "Seeded with SEED = 42" is a claim; a file listing
the interpreter, the library versions and the hash of each result JSON is
evidence, and it is what lets a reviewer tell a genuine reproduction from a
coincidence. The source hashes are the third leg: two runs agreeing on every
artefact hash while disagreeing on the code that wrote them is not a
reproduction, and without `sources` the manifest could not tell the difference.

Stages
  download      eight NHANES XPT files, four cycles  (skipped if present
                and structurally valid)
  dataprep      cohort audit and the shared 60/20/20 split, as a self-check
  identity_audit    audit the analyser identities and the rank deficiency
  ablation      the leakage ladder, the 2x2 decomposition, out-of-cycle replay
  benchmark     seven estimators + two trivial baselines, CIs, significance tests
  weighted      survey-design prevalence and weighted model metrics
  figures       the ten paper figures + results/importance.json
  app_models    trains and saves the L6 screening-demonstration models

Usage
  python src/run_all.py                  everything, in order
  python src/run_all.py --from ablation  resume from a stage
  python src/run_all.py --only figures   one stage
  python src/run_all.py --skip app_models --skip download
  python src/run_all.py --force-download re-download even if the XPTs exist
                                         (--only is refused alongside --skip
                                          or --from, not silently preferred)
  python src/run_all.py --list           show the stages and exit
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime

from download_data import CYCLES, cycle_files, problem
from paths import DATA, FIGURES, MODELS, RESULTS, ROOT, SRC

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_PATH = RESULTS / "run_all.log"
PREV_LOG_PATH = RESULTS / "run_all.prev.log"
MANIFEST_PATH = RESULTS / "run_manifest.json"
# Named once: versions() parses it for the dependency list and sources() hashes
# it, and those two must not be able to disagree about which file that is.
REQUIREMENTS = ROOT / "requirements.txt"
N_FIGURES = 10

# All eight XPTs, not just the primary cycle's two: ablation.py replays the
# ladder on 2011-12, 2013-14 and 2015-16, so a run that silently lacked those
# files would skip the only out-of-sample evidence in the study.
XPT_FILES = [p for cycle in CYCLES for p in cycle_files(cycle)]

# Each stage names the artefacts it is responsible for, so a stage that exits 0
# without writing its output is still reported as a failure.
STAGES = [
    {
        "name": "download",
        "script": "download_data.py",
        "what": f"fetch {len(XPT_FILES)} NHANES XPT files across "
                f"{len(CYCLES)} cycles",
        "outputs": XPT_FILES,
        "skip_if_outputs_exist": True,
        # download_data leaves an already-valid file alone, so this stage's
        # outputs can legitimately predate the run. Every other stage rewrites
        # everything it declares, and is now held to that.
        "outputs_may_predate_run": True,
    },
    {
        "name": "dataprep",
        "script": "dataprep.py",
        "what": "cohort, WHO labels, feature ladder, shared split (self-check)",
        "outputs": [],
    },
    {
        "name": "identity_audit",
        "script": "identity_audit.py",
        "what": "audit the analyser identities and the rank deficiency",
        "outputs": [RESULTS / "identity_audit.json"],
    },
    {
        "name": "ablation",
        "script": "ablation.py",
        "what": "leakage ladder, 2x2 decomposition, out-of-cycle replay",
        "outputs": [RESULTS / "ablation.json"],
    },
    {
        "name": "benchmark",
        "script": "benchmark.py",
        # "nine estimators, baselines" read as nine estimators PLUS baselines,
        # i.e. eleven models. There are nine entries in benchmark's `models` and
        # two of them ARE the baselines, so the two counts are stated separately.
        "what": "7 estimators + 2 trivial baselines, CIs, McNemar and bootstrap",
        "outputs": [RESULTS / "benchmark.json"],
    },
    {
        "name": "weighted",
        "script": "weighted.py",
        "what": "design-based prevalence and survey-weighted model metrics",
        "outputs": [RESULTS / "weighted.json"],
    },
    {
        "name": "figures",
        "script": "figures.py",
        "what": f"the {N_FIGURES} paper figures and importance.json",
        "outputs": [RESULTS / "importance.json"],
        "min_figures": N_FIGURES,
    },
    {
        "name": "app_models",
        "script": "anemia_app.py",
        "call": "train_models",
        "what": "train and save the L6 screening-demonstration models",
        "outputs": [MODELS / "anemia_binary.pkl", MODELS / "anemia_severity.pkl",
                    MODELS / "model_info.json"],
    },
]
STAGE_NAMES = [s["name"] for s in STAGES]

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    """
    Library versions, read from the installed distributions, not guessed.

    The names come from requirements.txt rather than from a list kept here. The
    list used to be typed out, which meant a new dependency was recorded in the
    manifest only if someone remembered to add it in two places - and the
    manifest is the thing a reader is supposed to trust about what produced the
    numbers. Parsing the pins also lets each entry carry the version that was
    ASKED for next to the one that is installed, and say whether they agree;
    recording only the installed version cannot distinguish a faithful
    environment from a drifted one.

    pyreadstat used to be on the typed list and always came back None: the XPT
    files are read by pandas' own SAS reader, pd.read_sas(format="xport"), and
    requirements.txt has never contained it. A null version in the manifest
    reads as a missing dependency rather than as an absent one.
    """
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, pinned = line.partition("==")
        name = name.strip()
        try:
            got = version(name)
        except PackageNotFoundError:
            got = None
        pinned = pinned.strip() or None
        out[name] = {"installed": got, "pinned": pinned,
                     "matches_pin": None if (got is None or pinned is None)
                                    else got == pinned}
    return out


def command(stage, force_download=False):
    """
    The subprocess for one stage.

    anemia_app.py is interactive, so its training path is called directly
    instead of running its menu. Everything else is just its own script, run
    with -u so its output interleaves correctly with ours in the log.
    """
    if "call" in stage:
        mod = stage["script"].removesuffix(".py")
        return [sys.executable, "-u", "-c",
                f"import {mod}; {mod}.{stage['call']}()"]
    cmd = [sys.executable, "-u", str(SRC / stage["script"])]
    # --force-download only ever skipped run_all's own existence check; nothing
    # was forwarded, and download_data skips any file already on disk, so the
    # flag could not re-download anything. It is passed through now.
    if force_download and stage.get("skip_if_outputs_exist"):
        cmd.append("--force")
    return cmd


def run_stage(stage, log, force_download=False):
    """Run one stage, echoing its output live and into the log. Returns a dict."""
    head = f"===== {stage['name']}  ({stage['what']}) ====="
    print(f"\n{head}")
    log.write(f"\n{head}\n")

    t0 = time.perf_counter()
    t_wall = time.time()          # for the freshness check below, not timing
    proc = subprocess.Popen(command(stage, force_download), cwd=str(SRC),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    secs = time.perf_counter() - t0

    # Existence was the only test, so a stage that failed after an earlier run
    # had already written its outputs still counted as "ok". Freshness is what
    # the check meant: an output has to be newer than the moment the stage
    # started. t_wall is captured before Popen for exactly this.
    may_predate = stage.get("outputs_may_predate_run")
    missing = []
    for out_p in stage["outputs"]:
        if not out_p.exists():
            missing.append(f"{out_p} (absent)")
        elif not may_predate and out_p.stat().st_mtime < t_wall - 1.0:
            missing.append(f"{out_p} (stale: not rewritten by this run)")

    # Figures are counted the same way. Counting every fig*.png in the folder
    # meant ten files left by an earlier run satisfied a stage that had just
    # produced none, and the count ignored the .pdf half of every figure.
    if stage.get("min_figures"):
        want = stage["min_figures"]
        for ext in ("png", "pdf"):
            fresh = [q for q in FIGURES.glob(f"fig*.{ext}")
                     if q.stat().st_mtime >= t_wall - 1.0]
            if len(fresh) < want:
                missing.append(f"{want} freshly written {ext.upper()} figures "
                               f"(found {len(fresh)})")

    ok = proc.returncode == 0 and not missing
    status = "ok" if ok else ("exit %d" % proc.returncode if proc.returncode
                              else "missing output")
    print(f"----- {stage['name']}: {status} in {secs:.1f}s")
    if missing:
        for m in missing:
            print(f"      missing: {m}")
    return {"name": stage["name"], "status": status, "ok": ok,
            "returncode": proc.returncode, "seconds": round(secs, 2),
            "missing": missing}


# Named, not silently omitted. Both of these change without the pipeline having
# run, so hashing them would make the manifest disagree with the disk for reasons
# that are not results. A reader can see the exclusions and their reasons here.
#
# model_info.json carries the wall clock it was trained at on purpose - it ships
# beside the deployable model - so its hash differs on every run by design.
#
# rejections.json is a running tally that anemia_app.py appends to every time
# somebody types a value into the interactive app. Hashing it meant the manifest
# was only valid until the next person opened the app: one session made every
# artefact check report a mismatch, on a counter that no table or figure reads.
#
# app_overview.png is the third of the same kind, and the only one that was
# invisible: it lands in figures/ beside the ten pipeline figures, so the folder
# holds an eleventh image that no figure table lists and no stage produces. It is
# written by anemia_app.show_graphs() - interactive menu option 3 - which is why
# the fig*.png glob never matched it and why it sat four hours older than every
# figure next to it. Named here so a reader who diffs figures/ against the
# manifest finds the explanation instead of an unaccounted file.
#
# The two logs are named for the same reason, having previously been left out by
# accident rather than by decision: artefacts() globs results/*.json, so a .log
# never matched it and the exclusion was a side effect of a file extension. That
# is the one thing this block exists to prevent - results/ held two files the
# manifest did not account for and gave a reader no way to tell whether that was
# deliberate. They cannot be hashed: run_all.log is being written at the moment
# the manifest is built, so its hash would describe a file that is still growing.
#
# The paper workflow also has two exported copies of its separately prepared
# flowchart in this directory. They are not produced by figures.py and are not
# used by any numerical result, so they are declared here rather than mistaken
# for unaccounted pipeline figures.
UNHASHED = {"model_info.json": "records its own training time by design",
            "rejections.json": "an interactive-app tally, appended outside the "
                               "pipeline; hashing it makes the manifest go stale "
                               "the first time anyone runs anemia_app.py",
            "app_overview.png": "written by anemia_app.show_graphs(), not by any "
                                "pipeline stage; it is an interactive-session "
                                "byproduct that happens to live in figures/",
            "fig1_flowchart_paper.png": "manuscript-only flowchart export; it is "
                                       "prepared outside figures.py and is not a "
                                       "numerical pipeline artefact",
            "fig1_flowchart_paper.pdf": "manuscript-only flowchart export; it is "
                                       "prepared outside figures.py and is not a "
                                       "numerical pipeline artefact",
            "run_all.log": "this run's own transcript, still open for writing "
                           "when the manifest is built, and it records wall "
                           "clocks and timings rather than results",
            "run_all.prev.log": "the previous run's transcript, kept so a "
                                "failure can be diffed against the last run "
                                "that worked; not a result of this one"}


def artefacts(t_start=None):
    """
    Hash everything a reader would compare against, in a stable order.

    The docstring above the module promises "a SHA-256 of every output", and
    this covered results/ and figures/ only: the two app_models pickles were
    declared as stage outputs and then left out of the manifest entirely. They
    are hashed now.

    Two further things are recorded per file rather than left to inference.

    `written_by_this_run` - the glob hashes what is on disk, which on a partial
    run means the manifest republishes hashes of files this run never touched,
    under the same key and in the same block as the ones it did. `complete_run:
    false` was the only signal, and it does not say WHICH artefacts are carried
    over. mtime against the moment the first stage started answers that per file.

    `undeclared` - the glob also admits anything else that happens to be a .json
    in results/ or a .pkl in models/. A stray from an abandoned branch was
    silently absorbed into the artefact count, so the manifest asserted
    provenance for a file no stage claims to write. Stage `outputs` plus
    figures/fig*.{png,pdf} is the declared set; the rest is flagged, not hidden.
    """
    files = (sorted(RESULTS.glob("*.json")) + sorted(FIGURES.glob("fig*.png"))
             + sorted(FIGURES.glob("fig*.pdf")) + sorted(MODELS.glob("*.pkl"))
             + sorted(MODELS.glob("*.json")))
    declared = {p.relative_to(ROOT).as_posix()
                for s in STAGES for p in s["outputs"]}
    out, undeclared, carried = {}, [], []
    for p in files:
        if p.name == MANIFEST_PATH.name or p.name in UNHASHED:
            continue
        key = p.relative_to(ROOT).as_posix()
        rec = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        if key not in declared and not (key.startswith("figures/")
                                        and p.name.startswith("fig")):
            rec["undeclared"] = True
            undeclared.append(key)
        if t_start is not None:
            fresh = p.stat().st_mtime >= t_start - 1.0
            rec["written_by_this_run"] = fresh
            if not fresh:
                carried.append(key)
        out[key] = rec
    return out, undeclared, carried


def inputs():
    """
    Hash the eight NHANES files the study reads.

    The manifest pinned every output and said nothing about what went in, so two
    runs could match on every artefact hash while having read different data.
    These eight files are the only ones in the project that nobody here wrote,
    which makes them the ones worth pinning.
    """
    out = {}
    for q in XPT_FILES:
        key = q.relative_to(ROOT).as_posix()
        out[key] = ({"absent": True} if not q.exists() else
                    {"bytes": q.stat().st_size, "sha256": sha256(q)})
    return out


def sources():
    """
    Hash the code, so the manifest says what produced the numbers and not only
    what the numbers are.

    Every hash in `inputs` and `artefacts` answers "did I read and write the same
    bytes as the reference run". Neither answers "did I run the same code", and
    that is the question this project has already got wrong once: the results were
    valid JSON, every stage had succeeded, and the manifest was four hours older
    than ablation.py, so figures.py was reading a key that had since been renamed.
    Two runs can agree on all 27 artefact hashes and disagree on the source that
    made them only if one of them is not a reproduction at all.

    src/*.py and requirements.txt, and nothing else. Not tests/, because a test
    edit does not change a single reported number and a manifest that goes stale
    for a reason unrelated to the results is a check people learn to ignore - the
    same argument that keeps rejections.json out of `artefacts`. requirements.txt
    is in because `libraries` records what was actually imported, which is the
    stronger fact, and a reproducer installs from the pin file: hashing it lets a
    reader see that the two agree instead of taking it on trust.

    Called BEFORE the first stage runs, not with the rest of the manifest at the
    end. Hashing at the end would record whatever the files say once the run is
    over, so a source edited while the long pipeline was in flight would be
    written down as the code that produced results it never touched - which is a
    more convincing version of exactly the desync this is here to catch.
    """
    out = {}
    for p in sorted(SRC.glob("*.py")) + [REQUIREMENTS]:
        if p.exists():
            out[p.relative_to(ROOT).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Run the whole study in order.")
    ap.add_argument("--from", dest="start", metavar="STAGE",
                    help="resume from this stage")
    ap.add_argument("--only", action="append", default=[], metavar="STAGE",
                    help="run only this stage (repeatable)")
    ap.add_argument("--skip", action="append", default=[], metavar="STAGE",
                    help="skip this stage (repeatable)")
    ap.add_argument("--force-download", action="store_true",
                    help="re-download the XPT files even if they are present")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()
    for name in a.only + a.skip + ([a.start] if a.start else []):
        if name not in STAGE_NAMES:
            ap.error(f"unknown stage {name!r}; choose from {STAGE_NAMES}")
    return a


def select(args):
    """
    Which stages to run. --only wins, and now says so instead of pretending.

    --only used to return early and discard --skip and --from without a word,
    so `--only figures --skip figures` ran figures. Combining them is a
    contradiction, not a preference, so it is refused.

    An empty selection is refused too. `all([])` is True and `len([]) ==
    len([])` is True, so `--from app_models --skip app_models` ran nothing,
    reported "the selected stages succeeded", wrote a manifest full of the
    previous run's hashes and exited 0. Nothing to do is not the same as
    everything worked.
    """
    stages = STAGES
    if args.only:
        if args.skip or args.start:
            raise SystemExit("run_all: --only cannot be combined with --skip "
                             "or --from; --only already names the selection.")
        chosen = [s for s in stages if s["name"] in args.only]
    else:
        if args.start:
            stages = stages[STAGE_NAMES.index(args.start):]
        chosen = [s for s in stages if s["name"] not in args.skip]
    if not chosen:
        raise SystemExit("run_all: that selection is empty, so there is nothing "
                         "to run and nothing to verify. Widen --from/--skip or "
                         "name a stage with --only.")
    return chosen


def main():
    args = parse_args()
    if args.list:
        for s in STAGES:
            print(f"  {s['name']:<15} {s['what']}")
        return 0

    chosen = select(args)
    # Before the first stage, not with the rest of the manifest at the end. See
    # sources(): hashing afterwards would record a file edited mid-run as the code
    # that produced results it never touched.
    src_before = sources()
    print(f"anemia_detection - full reproduction, {len(chosen)} stage(s)")
    print(f"root   {ROOT}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"log    {LOG_PATH}")
    print(f"source {len(src_before)} files hashed before the first stage")

    # One generation is kept. The log was opened "w" on every run, so a failing
    # run destroyed the output of the last one that had worked - the single most
    # useful thing to compare a failure against.
    if LOG_PATH.exists():
        LOG_PATH.replace(PREV_LOG_PATH)

    results = []
    t_run_start = time.time()
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"run_all {datetime.now().isoformat(timespec='seconds')}\n")
        for stage in chosen:
            if stage.get("skip_if_outputs_exist") and not args.force_download:
                # Existence was the whole test, so an interrupted download left
                # a short XPT that skipped this stage forever and then read as a
                # smaller cohort further down. Same structural check
                # download_data applies before it accepts a file.
                damaged = [(q, problem(q)) for q in stage["outputs"]]
                damaged = [(q, why) for q, why in damaged if why]
                if not damaged:
                    print(f"\n===== {stage['name']}: skipped, {DATA.name}/ "
                          f"already has all {len(stage['outputs'])} files and "
                          f"each one is structurally valid")
                    results.append({"name": stage["name"], "status": "skipped",
                                    "ok": True, "seconds": 0.0, "missing": []})
                    continue
                print(f"\n===== {stage['name']}: not skipped, "
                      f"{len(damaged)} of {len(stage['outputs'])} files need "
                      f"fetching")
                for q, why in damaged:
                    print(f"      {q.name}: {why}")
            r = run_stage(stage, log, force_download=args.force_download)
            results.append(r)
            if not r["ok"]:
                print(f"\nSTOPPED at {stage['name']}. Nothing after it ran, so "
                      f"the results directory is now inconsistent -\n"
                      f"fix the error and re-run with --from {stage['name']}.")
                break

    # ------------------------------------------------------------ summary ----
    # Collected rather than printed straight out: the summary, the manifest path
    # and the verdict all used to go to the terminal only, so run_all.log ended
    # at the last line of the last stage and never said whether the run passed.
    say = []
    say.append("\nSUMMARY")
    say.append(f"  {'stage':<15}{'status':<16}{'seconds':>9}")
    for r in results:
        say.append(f"  {r['name']:<15}{r['status']:<16}{r['seconds']:>9.1f}")
    total = sum(r["seconds"] for r in results)
    say.append(f"  {'total':<15}{'':<16}{total:>9.1f}")

    stages_ok = all(r["ok"] for r in results) and len(results) == len(chosen)

    # Hashed again now the run is over. Equal is the ordinary case and says the
    # code did not move under the run; unequal names the files, and is the one
    # failure mode a start-of-run hash would otherwise hide - the manifest would
    # look clean while pointing at source that no longer exists.
    src_after = sources()
    edited = sorted(k for k in set(src_before) | set(src_after)
                    if src_before.get(k) != src_after.get(k))
    if edited:
        say.append(f"\n  WARNING: edited while the run was in flight: {edited}")
        say.append("  The results below were produced by the source hashed at the "
                   "start, which is no longer what is on disk. Re-run.")

    arte, undeclared, carried = artefacts(t_run_start)
    if undeclared:
        say.append(f"\n  WARNING: hashed but declared by no stage: {undeclared}")
        say.append("  These are in `artefacts` with undeclared: true. A file no "
                   "stage claims to write has no provenance to record.")
    if carried:
        say.append(f"\n  NOTE: not written by this run, carried over from an "
                   f"earlier one: {carried}")
        say.append("  Each is flagged written_by_this_run: false in the manifest.")

    full_selection = not args.only and not args.skip and not args.start
    # `edited` belongs in `ok`, not only in a warning. It used to print the
    # warning above and then fall through to "Every stage succeeded, so every
    # table and figure in the paper comes from the artefacts hashed in the
    # manifest" - which is the exact claim the warning had just withdrawn - and
    # exit 0. A run whose source moved underneath it is not a reproduction.
    ok = stages_ok and not edited
    # A full selection that still carries artefacts forward, or that hashed
    # something no stage declares, is not a complete run either: the manifest
    # would be asserting provenance over files this run did not produce.
    complete = ok and full_selection and not carried and not undeclared

    import dataprep as dp                      # for the seed, after the run
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "complete_run": complete,
        "seed": dp.SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": versions(),
        "stages": results,
        "total_seconds": round(total, 2),
        "sources": src_before,
        "sources_unchanged_during_run": not edited,
        "sources_edited_during_run": edited,
        "inputs": inputs(),
        "artefacts": arte,
        "artefacts_not_hashed": UNHASHED,
        "artefacts_undeclared": undeclared,
        "artefacts_carried_from_an_earlier_run": carried,
        "selection_was_the_full_pipeline": full_selection,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, allow_nan=False)
    say.append(f"\nwrote {MANIFEST_PATH}")
    say.append(f"  {len(manifest['inputs'])} input files, "
               f"{len(manifest['sources'])} source files and "
               f"{len(manifest['artefacts'])} artefacts hashed")
    say.append(f"  {len(UNHASHED)} files excluded from hashing, each with a "
               f"reason: {', '.join(sorted(UNHASHED))}")

    if manifest["complete_run"]:
        say.append("\nEvery stage succeeded, so every table and figure in the "
                   "paper")
        say.append("comes from the artefacts hashed in the manifest.")
    elif ok:
        say.append("\nThe selected stages succeeded, but this is not a complete "
                   "run, so the")
        say.append("manifest is marked complete_run: false. Reason: "
                   + ("a partial selection" if not full_selection else
                      "artefacts this run did not write" if carried else
                      "artefacts no stage declares"))
    else:
        failed = [r["name"] for r in results if not r["ok"]]
        if edited and not failed:
            say.append("\nFAILED: every stage exited cleanly, but the source "
                       "moved while the run was in")
            say.append("flight, so the manifest's `sources` block does not "
                       "describe the code on disk. Re-run.")
        else:
            say.append(f"\nFAILED at {', '.join(failed) or 'an unrun stage'}. The "
                       f"results directory is inconsistent;")
            say.append("the manifest was still written so the failure is on "
                       "record, with complete_run false.")

    for line in say:
        print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write("\n".join(say) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
