"""
run_all's stage selection, which decides what a "reproduction" actually contains.

This is the one part of run_all.py that can be tested without running the
pipeline, and it is the part whose failures are silent. The manifest is the
document the paper points at for provenance, and it is written by the same
function regardless of how few stages ran - so a selection bug does not crash,
it publishes.

Both defects tested here were real and both exited 0:

  --only figures --skip figures      ran figures. --only returned early and
                                     discarded --skip and --from without a word,
                                     so a contradiction read as a preference.

  --from app_models --skip app_models
                                     ran NOTHING, printed "the selected stages
                                     succeeded", re-hashed the previous run's
                                     artefacts into a fresh manifest and exited
                                     0. all([]) is True and len([]) == len([])
                                     is True, so every completeness check passed
                                     on an empty selection.

No stage is executed by anything in this file.
"""

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def ra():
    import run_all
    return run_all


class Args:
    """The five fields select() reads, defaulted to a full run."""

    def __init__(self, only=(), skip=(), start=None):
        self.only = list(only)
        self.skip = list(skip)
        self.start = start
        self.list = False
        self.force_download = False


def names(chosen):
    return [s["name"] for s in chosen]


def test_the_default_selection_is_every_declared_stage_in_order(ra):
    """
    Order matters as much as membership: figures read what weighted writes.

    STAGE_NAMES is derived from STAGES rather than typed beside it, and this is
    the check that the derivation is what select() hands back.
    """
    assert names(ra.select(Args())) == ra.STAGE_NAMES
    assert ra.STAGE_NAMES == ["download", "dataprep", "identity_audit",
                              "ablation", "benchmark", "weighted", "figures",
                              "app_models"]


def test_only_names_the_selection_exactly(ra):
    assert names(ra.select(Args(only=["figures"]))) == ["figures"]
    assert names(ra.select(Args(only=["weighted", "figures"]))) == ["weighted",
                                                                   "figures"]
    # Declaration order, not the order they were typed on the command line.
    assert names(ra.select(Args(only=["figures", "weighted"]))) == ["weighted",
                                                                   "figures"]


def test_from_starts_at_a_stage_and_keeps_everything_after_it(ra):
    got = names(ra.select(Args(start="benchmark")))
    assert got == ["benchmark", "weighted", "figures", "app_models"]
    assert names(ra.select(Args(start="download"))) == ra.STAGE_NAMES


def test_skip_removes_stages_without_reordering_the_rest(ra):
    got = names(ra.select(Args(skip=["download", "figures"])))
    assert got == ["dataprep", "identity_audit", "ablation", "benchmark",
                   "weighted", "app_models"]


def test_from_and_skip_compose(ra):
    got = names(ra.select(Args(start="ablation", skip=["benchmark"])))
    assert got == ["ablation", "weighted", "figures", "app_models"]


@pytest.mark.parametrize("kw", [{"skip": ["figures"]},
                                {"start": "weighted"},
                                {"skip": ["download"], "start": "ablation"}])
def test_only_refuses_to_be_combined_rather_than_silently_winning(ra, kw):
    """
    The refusal is the fix. `--only figures --skip figures` used to run figures.

    Silently discarding --skip is worse than either honouring it or refusing it,
    because the terminal then shows a stage running that the operator explicitly
    asked to skip, and the manifest records it as a deliberate selection.
    """
    with pytest.raises(SystemExit) as e:
        ra.select(Args(only=["figures"], **kw))
    msg = str(e.value)
    assert "--only" in msg and ("--skip" in msg or "--from" in msg), msg


@pytest.mark.parametrize("kw", [
    {"start": "app_models", "skip": ["app_models"]},
    {"skip": ["download", "dataprep", "identity_audit", "ablation",
              "benchmark", "weighted", "figures", "app_models"]},
    {"only": ["__no_such_stage__"]},
])
def test_an_empty_selection_is_refused_instead_of_reported_as_a_success(ra, kw):
    """
    Nothing to do is not the same as everything worked.

    The third case is the one that still gets through argument parsing: an
    unknown name is rejected by parse_args, but select() is also called directly
    (here, and by anything that builds Args itself), so the emptiness check has
    to live in select() and not only in the parser.
    """
    with pytest.raises(SystemExit) as e:
        ra.select(Args(**kw))
    assert "empty" in str(e.value).lower(), str(e.value)


def test_every_stage_declares_outputs_and_they_are_all_inside_the_project(ra):
    """
    `outputs` is what run_stage checks for freshness and what sources() calls
    declared, so a stage with no outputs is a stage that cannot fail its own
    check - and an output outside ROOT would be hashed into the manifest under a
    path a reader cannot resolve relative to the repository.

    dataprep is the single exception and it is named here rather than waved
    through by an `or`: it writes nothing, so it is the one stage that can only
    fail by raising. Its own --list line has to say "self-check", because the
    exception is only defensible while a reader can see it in `--list`. Any
    second stage that declares nothing is a stage whose failure mode is exiting 0.
    """
    silent = [s["name"] for s in ra.STAGES
              if not s["outputs"] and not s.get("min_figures")]
    assert silent == ["dataprep"], (
        f"these stages cannot fail their own output check: {silent}")
    assert "self-check" in dict((s["name"], s["what"])
                                for s in ra.STAGES)["dataprep"]
    for s in ra.STAGES:
        for p in s["outputs"]:
            assert ra.ROOT in p.parents, f"{s['name']}: {p} is outside the project"
        assert s["what"], f"{s['name']} has no --list description"


def test_the_declared_output_set_matches_the_artefacts_the_manifest_hashes(
        ra, manifest_json):
    """
    The other half of `undeclared`: every declared output must have been hashed.

    sources() flags an artefact no stage claims to write. Nothing flagged the
    reverse - a stage output that the artefact glob does not reach - and the two
    app_models pickles were exactly that for as long as the manifest existed:
    declared as stage outputs, then left out of the hash block entirely.

    `inputs` counts as hashed. download's declared outputs are the eight XPT
    files, and those are recorded in the manifest's inputs block rather than its
    artefacts block - they are the study's raw material, hashed under the name a
    reader would look for them by. Hashed somewhere with a reason is the contract;
    hashed in a particular block is not.
    """
    declared = {p.relative_to(ra.ROOT).as_posix()
                for s in ra.STAGES for p in s["outputs"]}
    hashed = set(manifest_json["artefacts"]) | set(manifest_json["inputs"])
    excluded = set(manifest_json["artefacts_not_hashed"])
    missing = sorted(d for d in declared
                     if d not in hashed and d.rsplit("/", 1)[-1] not in excluded)
    assert not missing, f"declared as stage outputs but never hashed: {missing}"
    assert "models/anemia_binary.pkl" in hashed
    assert "models/anemia_severity.pkl" in hashed
    # The eight XPT files are declared by download and hashed as inputs, so the
    # union above must not be doing that work on its own for the results/ JSONs.
    for name in ("results/ablation.json", "results/benchmark.json",
                 "results/weighted.json", "results/identity_audit.json",
                 "results/importance.json"):
        assert name in manifest_json["artefacts"], name


def test_the_manifest_records_whether_the_selection_was_the_whole_pipeline(
        ra, manifest_json):
    """
    complete_run without this key cannot distinguish two different situations.

    A partial run that succeeded and a full run that succeeded both used to write
    "every stage succeeded", and the artefacts in results/ after the first are a
    mixture of two runs. The three keys are read together: a full selection, no
    artefact carried over from an earlier run, and nothing in the directories
    that no stage claims to write.
    """
    for key in ("selection_was_the_full_pipeline", "artefacts_undeclared",
                "artefacts_carried_from_an_earlier_run"):
        assert key in manifest_json, f"{key} is missing from the manifest"
    assert manifest_json["artefacts_undeclared"] == [], (
        "results/ or models/ holds a file no stage writes: "
        f"{manifest_json['artefacts_undeclared']}")
    assert manifest_json["artefacts_carried_from_an_earlier_run"] == [], (
        "these were not written by the run that wrote the manifest: "
        f"{manifest_json['artefacts_carried_from_an_earlier_run']}")
    assert manifest_json["selection_was_the_full_pipeline"] is True
    assert manifest_json["complete_run"] is True
    for rec in manifest_json["artefacts"].values():
        assert rec["written_by_this_run"] is True


def test_the_unhashed_names_are_not_also_declared_stage_outputs(ra):
    """
    An exclusion that names a declared output would hide a real gap behind a
    justification. The seven exclusions are files that change without the pipeline
    running (the app's rejection tally, a training timestamp, the two logs) plus
    one image no stage produces; only one of them is also a stage's output.

    model_info.json is that one, and it is the deliberate exception: app_models
    writes it, and it records the wall clock at which it was written, so its hash
    changes on a re-run that produced identical models. Every other exclusion has
    to stay outside the declared set, because an exclusion that quietly names a
    real output turns a gap in the manifest into a justified-looking absence.
    """
    declared = {p.name for s in ra.STAGES for p in s["outputs"]}
    overlap = sorted(set(ra.UNHASHED) & declared)
    assert overlap == ["model_info.json"], overlap
    # The reason has to be the time-varying one. Any other wording here would be
    # excusing a declared output on grounds that do not apply to it.
    reason = ra.UNHASHED["model_info.json"].lower()
    assert any(w in reason for w in ("clock", "time", "trained at")), reason


def test_every_unhashed_name_is_given_a_reason_a_reader_can_evaluate(ra):
    """
    UNHASHED is a dict rather than a list so that no name can be excluded without
    an argument attached. A one-word reason would satisfy the type and defeat the
    purpose, so the length is checked as well as the presence.
    """
    assert len(ra.UNHASHED) == 7, sorted(ra.UNHASHED)
    for name, reason in ra.UNHASHED.items():
        assert isinstance(reason, str) and len(reason) > 30, (name, reason)
