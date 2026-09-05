"""
The guard against the state this project was actually found in: source newer than
results.

Nothing in a pipeline run can catch this. Every stage succeeded, every JSON was
valid, every figure was written - and figures.py was reading a key that ablation.py
had renamed, because the last full run predated the rename. The manifest said so,
and nobody was looking at the manifest.

So this file checks three things a single run cannot:

  1. every source file still hashes to what the manifest recorded, i.e. the
     results on disk were produced by the code on disk;
  2. every artefact still hashes to what the manifest recorded;
  3. every key the figures dereference is present in the JSON they read it from,
     spelled out here so a rename breaks a test instead of a figure.

(3) is deliberately a hand-written list rather than something derived from
figures.py. Deriving it would make the test agree with the source by construction,
which is the one thing it must not do.
"""

import datetime
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent


def deref(blob, path, where):
    """blob['a']['b'][0]['c'] from 'a.b[0].c', with a readable failure."""
    cur = blob
    for step in path.split("."):
        while step.endswith("]"):
            step, _, idx = step[:-1].rpartition("[")
            if step:
                assert isinstance(cur, dict) and step in cur, f"{where}: no {path} (at {step})"
                cur = cur[step]
            assert isinstance(cur, list), f"{where}: {path} is not a list at {step or idx}"
            assert len(cur) > int(idx), f"{where}: {path} has only {len(cur)} entries"
            cur = cur[int(idx)]
            step = ""
        if step:
            assert isinstance(cur, dict) and step in cur, f"{where}: no {path} (at {step})"
            cur = cur[step]
    return cur


# ---------------------------------------------------------------- 1. freshness

def test_every_source_file_still_hashes_to_what_the_manifest_recorded(manifest_json):
    """
    The exact desync that was reported: manifest 07:36, sources edited until 09:42.

    This was an mtime comparison first, which is what the manifest could support at
    the time - it hashed the data and the outputs and said nothing about the code.
    Mtimes are wrong in both directions: touching a file without changing a
    character failed the test, and a file restored from a copy with an older
    timestamp passed it while holding different code. Hashes have neither problem,
    so the manifest now records src/*.py and requirements.txt and this compares
    against that.

    A new source file that no run has seen fails here too, under `unknown` - a
    tenth script in src/ that the pipeline never called is a real desync, not a
    bookkeeping gap.
    """
    recorded = manifest_json["sources"]
    changed, gone = [], []
    for rel, rec in sorted(recorded.items()):
        p = ROOT / rel
        if not p.exists():
            gone.append(rel)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            changed.append(rel)
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").glob("*.py")}
    unknown = sorted(on_disk - set(recorded))
    generated = datetime.datetime.fromisoformat(manifest_json["generated"])
    assert not changed, (
        f"manifest generated {generated:%Y-%m-%d %H:%M:%S}, but these have been "
        f"edited since: {changed}. Run python src/run_all.py")
    assert not gone, f"the manifest names source that is not there: {gone}"
    assert not unknown, f"src/ holds code no run has recorded: {unknown}"


def test_the_manifest_hashes_the_scripts_that_produce_the_numbers(manifest_json):
    """
    Hashing *something* is not the claim; hashing the code on the critical path is.

    A `sources` block that quietly stopped covering ablation.py would leave the
    test above passing on whatever it did still cover. requirements.txt is in the
    list because a reproducer installs from it.

    All eleven entries are named, not seven of them. The short list was a subset
    check: it left download_data.py, paths.py and anemia_app.py unnamed, so the
    manifest could stop hashing the downloader - the one file that decides which
    bytes the whole study reads - and this test would still pass. The count is
    asserted too, because a name list with no total is satisfied by a superset.
    """
    recorded = manifest_json["sources"]
    expected = {"src/ablation.py", "src/anemia_app.py", "src/benchmark.py",
                "src/dataprep.py", "src/download_data.py", "src/figures.py",
                "src/identity_audit.py", "src/paths.py", "src/run_all.py",
                "src/weighted.py", "requirements.txt"}
    assert set(recorded) == expected, sorted(set(recorded) ^ expected)
    for name in sorted(expected):
        assert recorded[name]["bytes"] > 0, name
        assert len(recorded[name]["sha256"]) == 64, name
        assert set(recorded[name]["sha256"]) <= set("0123456789abcdef"), name


def test_the_source_did_not_move_while_the_run_was_in_flight(manifest_json):
    """
    The hashes are taken before the first stage, so this is the other half of that.

    Taking them at the end would record a mid-run edit as the code that produced
    results it never touched - a desync that reads as clean. run_all hashes again
    afterwards and writes down the comparison; a manifest that admits its own
    sources moved is not evidence of anything.
    """
    assert manifest_json["sources_unchanged_during_run"] is True, (
        f"edited mid-run: {manifest_json['sources_edited_during_run']}")
    assert manifest_json["sources_edited_during_run"] == []


def test_every_stage_succeeded(manifest_json):
    """
    The count is asserted first, because "every stage in the list succeeded" is
    true of an empty list.

    run_all declares eight stages and the manifest has to carry all eight in
    order. A manifest holding seven is a run with a silently dropped stage in it,
    and that reads as clean: run_all only breaks the loop on a stage that FAILED,
    so a stage removed from the list leaves every remaining record ok. `download`
    is normally "skipped" rather than "ok" - the XPT files are already on disk and
    structurally valid - and run_all records that as ok: True on purpose, so the
    status is checked through the ok flag rather than against the word.
    """
    assert manifest_json["selection_was_the_full_pipeline"] is True, (
        "these artefacts came from a partial run; results/ is a mix of runs")
    names = [s["name"] for s in manifest_json["stages"]]
    assert names == ["download", "dataprep", "identity_audit", "ablation",
                     "benchmark", "weighted", "figures", "app_models"], names
    for stage in manifest_json["stages"]:
        assert stage["ok"] is True, f"{stage['name']}: {stage['status']}"
        assert not stage.get("missing"), f"{stage['name']} missing {stage['missing']}"
    # Last, so that a failing stage above reports itself by name rather than
    # arriving here as an opaque False.
    assert manifest_json["complete_run"] is True


def test_every_artefact_still_hashes_to_what_the_manifest_recorded(manifest_json):
    """
    27 is asserted before the loop: 5 results JSON, 10 PNG, 10 PDF, 2 PKL.

    Without the count this test passes on an empty `artefacts` block - a manifest
    that hashed nothing satisfies "nothing it hashed has changed". That is not a
    hypothetical: the block is built by globbing four directories, so a path
    constant that stopped resolving would empty it silently and every assertion
    below would still hold.
    """
    art = manifest_json["artefacts"]
    assert len(art) == 27, f"{len(art)} artefacts hashed, expected 27: {sorted(art)}"
    bad, gone = [], []
    for rel, rec in sorted(art.items()):
        p = ROOT / rel
        if not p.exists():
            gone.append(rel)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            bad.append(rel)
    assert not gone, f"manifest lists files that are not there: {gone}"
    assert not bad, f"changed since the run that produced the manifest: {bad}"


def test_every_input_file_still_hashes_to_what_the_manifest_recorded(manifest_json):
    """
    Eight XPT files, and at least the four the primary cohort is built from present.

    The `absent` skip is what makes the count necessary. A cycle whose files never
    downloaded is recorded rather than hashed, so a run in which every input was
    absent would iterate eight times and assert nothing at all.
    """
    inputs = manifest_json["inputs"]
    assert len(inputs) == 8, f"{len(inputs)} inputs recorded: {sorted(inputs)}"
    checked = 0
    for rel, rec in sorted(inputs.items()):
        if rec.get("absent"):
            continue
        p = ROOT / rel
        assert p.exists(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"], rel
        checked += 1
    assert checked == 8, f"only {checked} of 8 input files were hashed"


def test_the_unhashed_exclusions_are_named_and_justified(manifest_json):
    """
    Seven files are excluded from hashing, and every exclusion has to stay visible.

    rejections.json is the one that matters most: anemia_app.py appends to it every
    time somebody uses the interactive app, so while it was hashed the manifest
    stopped matching the disk the first time anyone opened the app - a provenance
    check that fails for a reason unrelated to provenance is a check people learn
    to ignore.

    app_overview.png is the one that was invisible rather than wrong. It sits in
    figures/ beside the ten pipeline figures without being one of them, and the
    fig*.png glob never matched it, so the folder held an unaccounted eleventh
    image. Listing it is the whole fix: a reader who diffs figures/ against the
    manifest now finds a reason instead of a gap.

    The two logs are the newest entries and were previously excluded by accident
    rather than by decision: the artefact walk globbed *.json and *.pkl, so a
    .log file fell outside it on file extension. run_all.log is still open for
    writing when the manifest is built - hashing it would hash a truncated copy
    of itself - and run_all.prev.log belongs to the previous run by definition.
    Both now say so out loud, because "not hashed because the glob missed it" and
    "not hashed because it cannot meaningfully be hashed" read identically from
    the manifest and only one of them is a decision.
    """
    excluded = manifest_json["artefacts_not_hashed"]
    assert set(excluded) == {"model_info.json", "rejections.json",
                             "app_overview.png",
                             "fig1_flowchart_paper.png",
                             "fig1_flowchart_paper.pdf",
                             "run_all.log", "run_all.prev.log"}, excluded
    for name, reason in excluded.items():
        assert len(reason) > 20, f"{name} is excluded without a reason"
    hashed = {Path(k).name for k in manifest_json["artefacts"]}
    assert not (hashed & set(excluded)), "a file is both hashed and declared unhashed"


def test_the_only_unlisted_image_in_figures_is_the_declared_one(manifest_json):
    """
    Every image in figures/ is either a manifest artefact or a named exclusion.

    Without this, the exclusion list documents the one stray file that was found
    and says nothing about the next one. A twelfth image appearing there - a
    leftover draft, a figure from a branch that was never re-run - fails here.

    Images only, by suffix: .gitkeep is what makes the empty directory exist in the
    first place, and a rule that flags repository plumbing as an unaccounted result
    is a rule that gets an exception added to it and then ignored.
    """
    listed = {Path(k).name for k in manifest_json["artefacts"]
              if k.startswith("figures/")}
    excluded = set(manifest_json["artefacts_not_hashed"])
    stray = sorted(p.name for p in (ROOT / "figures").iterdir()
                   if p.is_file() and p.suffix.lower() in (".png", ".pdf")
                   and p.name not in listed and p.name not in excluded)
    assert not stray, f"figures/ holds images the manifest does not account for: {stray}"


def test_the_manifest_pins_the_library_versions_that_produced_it(manifest_json):
    """A hash is provenance only if it says what read the data."""
    libs = manifest_json["libraries"]
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost"):
        assert name in libs and libs[name]["installed"], name
    # versions() now reads the names out of requirements.txt and records the
    # pinned version beside the installed one, so the agreement is checked by
    # the recorder and re-checked here. The old form of this test rebuilt the
    # "name==version" string and searched the pin file for it, which passed
    # whenever a dependency was missing from BOTH the typed name list and the
    # environment - the one case worth catching.
    req_names = {line.split("#", 1)[0].split("==")[0].strip()
                 for line in (ROOT / "requirements.txt").read_text().splitlines()
                 if line.split("#", 1)[0].strip()}
    assert set(libs) == req_names, (
        f"manifest libraries and requirements.txt disagree: "
        f"{sorted(set(libs) ^ req_names)}")
    for name, rec in libs.items():
        assert rec["installed"] is not None, f"{name} is pinned but not installed"
        assert rec["pinned"] is not None, f"{name} is recorded but not pinned"
        assert rec["matches_pin"], (
            f"the run used {name} {rec['installed']}, "
            f"requirements.txt pins {rec['pinned']}")


# ------------------------------------------------- 2. keys the figures read

ABLATION_KEYS = [
    "cohort.n_analysis", "n_train", "n_val", "n_test", "n_test_clusters",
    "majority_baseline_binary", "pr_auc_no_skill",
    "rungs.L6_identity_free_demo.features",
    "rungs.L6_identity_free_demo.binary.test_accuracy",
    "rungs.L6_identity_free_demo.binary.test_accuracy_ci95",
    "rungs.L6_identity_free_demo.binary.pr_auc",
    "rungs.L6_identity_free_demo.binary.feature_importance_mdi",
    "rungs.L6_identity_free_demo.recoverability.selected",
    "rungs.L6_identity_free_demo.recoverability.best_on_test",
    "contrasts[0].contrast", "contrasts[0].delta_accuracy", "contrasts[0].ci95",
    "contrasts[0].p_holm", "contrasts[0].significant_holm_05",
    "variance_scales.median_se_ratio_raowu_over_row",
    "decomposition.effects_on_lift_scale", "decomposition.cells",
    "out_of_cycle.2011-2012.n", "out_of_cycle.2011-2012.prevalence",
    "out_of_cycle.2011-2012.rungs.L1_paper_leaky.lift_points",
    "complete_case_sensitivity.answer_primary_cycle",
    "complete_case_sensitivity.rungs",
    "cluster_holdout.absolute_optimism_mean",
    # The cohort audit's WHO-threshold block. This is where the manuscript's whole
    # label criticism lives - 56.72% of the cohort has a cut-off that is not 12.0 -
    # and it reached an artefact only recently: describe() printed it to stdout for
    # every version before that, so the number the argument rests on existed in a
    # console log. fig1's flowchart now reads thresholds_gdl instead of carrying a
    # re-typed literal, which makes a rename here a broken figure.
    "cohort.who_thresholds.thresholds_gdl",
    "cohort.who_thresholds.rows_per_threshold",
    "cohort.who_thresholds.rows_whose_cutoff_is_not_12",
    "cohort.who_thresholds.pct_whose_cutoff_is_not_12",
    "cohort.who_thresholds.binary_labels_that_flip_under_a_flat_12_cut",
    "cohort.who_thresholds.pct_binary_labels_that_flip",
    "cohort.attrition_profile", "cohort.pregnancy", "cohort.integrity",
    # The other direction: blocks this file publishes that nothing outside
    # ablation.py referenced. None of them is dead - they are the verdicts a
    # reviewer reads - but nothing pinned their names, so a rename would have
    # stranded the paper's own conclusions with every test still green.
    "majority_baseline_severity", "contrasts_sign_convention",
    "surviving_contrasts", "null_contrasts", "contrast_verdict",
    "out_of_cycle_ordering.L1_paper_leaky > L4_identity_free.gap_by_cycle",
    "out_of_cycle_ordering.L1_paper_leaky > L4_identity_free.holds_in_all_cycles",
    "out_of_cycle_ordering.L1_paper_leaky > L4_identity_free.smallest_margin",
    "out_of_cycle_ordering.L1_paper_leaky > L4_identity_free.smallest_margin_cycle",
    "train_size_sensitivity.conclusions_that_flip_at_05",
    "train_size_sensitivity.flip_criterion",
    "train_size_sensitivity.flips_if_60pct_arm_uses_holm",
    "tracking.all_rungs.spearman_rho", "tracking.monotonicity_violations",
    "tracking.claim",
]

BENCHMARK_KEYS = [
    "no_information_rate", "selected_model", "like_for_like_model",
    "selection_metric",
    "binary.L1_paper_leaky.models.rf.pr_auc",
    "binary.L1_paper_leaky.models.rf.calibration.brier",
    "binary.L1_paper_leaky.models.rf.calibration.bin_observed",
    "binary.L1_paper_leaky.models.rf.calibration.bin_predicted",
    "binary.L1_paper_leaky.models.rf.confusion_matrix",
    "binary.L1_paper_leaky.models.rf.thresholds.youden",
    "binary.L1_paper_leaky.models.rf.test_accuracy_design_se",
    "binary.L1_paper_leaky.models.rf.test_accuracy_row_se",
    "significance.family", "significance.correction",
    # The reverse direction here too. selection_table is the answer to R6-2 - it
    # is the evidence that Random Forest was not simply asserted - and neither it
    # nor the two spread blocks nor the prose answers were named anywhere outside
    # the script that writes them.
    "selection_table.rf", "selection_table.histgb", "selection_table.logreg",
    "accuracy_spread_across_models.L6_identity_free_demo",
    "identity_gap_across_estimators.n_estimators",
    "identity_gap_across_estimators.n_positive_accuracy",
    "identity_gap_across_estimators.n_positive_pr_auc",
    "identity_gap_across_estimators.mean_gap_accuracy",
    "identity_gap_across_estimators.mean_gap_pr_auc",
    "answers.why_random_forest", "answers.spread_vs_rung",
    "significance.selected_vs_like_for_like.accuracy",
    "significance.selected_vs_like_for_like.pr_auc",
    "significance.selected_vs_like_for_like.meaning",
]

WEIGHTED_KEYS = [
    "weight_variable", "weight_source_column",
    "design.n_strata", "design.total_psus", "design.df",
    "design.population_represented", "design.weight_cv",
    "prevalence.Anemia.prop_weighted", "prevalence.Anemia.ci95_logit",
    "prevalence.Anemia.design_effect",
    "prevalence.Anemia.deff_unequal_weighting",
    "prevalence.Anemia.deff_clustering_residual",
    "prevalence.Anemia_hb12.prop_weighted",
    "subgroups.all.prop_weighted", "subgroups.pregnant.prop_weighted",
    "model_oof.metrics.accuracy", "model_oof.unweighted_inside_ci",
    "fit_weight_sensitivity.verdict",
    # The Holm block. This file had no correction at all while the other two
    # scripts corrected their families, so every verdict in it read a raw p; the
    # keys that carry the fix are the ones most worth pinning, because their
    # absence would silently restore the old behaviour for any reader who looks
    # up p_value instead. README quotes the accuracy contrast by name.
    "fit_weight_sensitivity.significance.family",
    "fit_weight_sensitivity.significance.correction",
    "fit_weight_sensitivity.paired_contrasts_weighted_minus_unweighted_fit"
    ".accuracy.p_value",
    "fit_weight_sensitivity.paired_contrasts_weighted_minus_unweighted_fit"
    ".accuracy.p_holm",
    "fit_weight_sensitivity.paired_contrasts_weighted_minus_unweighted_fit"
    ".accuracy.significant_holm_05",
    "fit_weight_sensitivity.paired_contrasts_weighted_minus_unweighted_fit"
    ".sensitivity_recall.p_holm",
    "fit_weight_sensitivity.paired_contrasts_weighted_minus_unweighted_fit"
    ".pr_auc.p_holm",
    # The disclosure that used to be three hand-copied literals.
    "model_oof.shared_cluster_optimism.available",
    "model_oof.shared_cluster_optimism.accuracy_points_mean",
    "model_oof.shared_cluster_optimism.accuracy_points_worst_case",
    "model_oof.shared_cluster_optimism.exceeds_ci_half_width",
    "model_oof.metrics.accuracy.ci95_wald",
]

# identity_audit.json had no entry here and no figure consumer, which made it the
# one artefact outside the mechanism this file exists to provide. That is the
# wrong artefact to leave uncovered: it is the whole evidence base for the
# leakage claim - the four arithmetic routes to hemoglobin, their WHO label
# agreement, the 31-subset search, the rank spectra and the rounding null - and
# the README quotes leaves out of it by name. A rename in identity_audit.py
# could silently strand every one of those numbers.
#
# Written against the CURRENT names on purpose. Two keys in the replication block
# were called log_dof and levels_dof and are now log_components_at_cutoff and
# levels_components_at_cutoff, because a component count at a 99.9% variance
# cutoff is not a degree of freedom and this file publishes the arithmetic
# degrees_of_freedom separately; a third, ranks.log_rank_at_cutoff, was a
# duplicate of ranks.logs.k and is gone. If a future rename reverts any of that,
# this list is where it fails.
IDENTITY_KEYS = [
    "primary_cycle", "replication_note",
    "primary.cycle", "primary.n",
    "primary.identities[0].identity", "primary.identities[0].mae",
    "primary.identities[0].median_rounding_bound",
    "primary.identities[0].rows_exceeding", "primary.identities[0].pct_exceeding",
    # the four HGB routes, by the exact expression strings the README quotes
    "primary.routes.cross_checked_against",
    "primary.routes.to_hgb.MCH*RBC/10.mae",
    "primary.routes.to_hgb.MCH*RBC/10.label_agreement_who",
    "primary.routes.to_hgb.MCH*RBC/10.needs",
    "primary.routes.to_hgb.MCHC*HCT/100.mae",
    "primary.routes.to_hgb.MCHC*MCV*RBC/1000.mae",
    "primary.routes.to_hgb.MCH*HCT/MCV.mae",
    "primary.routes.to_hct.MCV*RBC/10.mae",
    # the subset search and the smallest exact route
    "primary.subset_search.n_subsets", "primary.subset_search.n_exact",
    "primary.subset_search.exact_threshold_mae",
    "primary.subset_search.worst_exact_mae",
    "primary.subset_search.best_non_exact_mae",
    "primary.subset_search.gap_at_threshold",
    "primary.subset_search.rows[0].features",
    "primary.subset_search.rows[0].mae",
    "primary.subset_search.rows[0].label_agreement_who",
    "primary.subset_search.minimal_exact[0].features",
    "primary.subset_search.smallest_exact.features",
    "primary.subset_search.smallest_exact.k",
    "primary.subset_search.smallest_exact.mae",
    "primary.subset_search.smallest_exact.label_agreement_who",
    # the rank spectra, the arithmetic dof, and the rounding null
    "primary.ranks.variables", "primary.ranks.cumulative_variance_cutoff",
    "primary.ranks.standardisation",
    "primary.ranks.levels.k", "primary.ranks.levels.k_999",
    "primary.ranks.levels.k_9999", "primary.ranks.levels.variance_ratio",
    "primary.ranks.logs.k", "primary.ranks.logs.k_999",
    "primary.ranks.logs.k_9999", "primary.ranks.logs.variance_ratio",
    "primary.ranks.logs.trailing3_variance_pct",
    "primary.ranks.n_identities", "primary.ranks.degrees_of_freedom",
    "primary.ranks.levels_minus_logs", "primary.ranks.note",
    "primary.ranks.rounding_null.n_draws", "primary.ranks.rounding_null.seed",
    "primary.ranks.rounding_null.trailing3_variance_pct",
    "primary.ranks.rounding_null.trailing3_variance_pct_sd_across_draws",
    "primary.ranks.rounding_null.naive_understates_null_by",
    "primary.ranks.clean_rows.n_excluded",
    "primary.ranks.clean_rows.trailing3_variance_pct",
    "primary.ranks.clean_rows.random_exclusion_control.trailing3_variance_pct",
    "primary.ranks.trailing_ratio_observed_to_null",
    "primary.ranks.trailing_ratio_clean_to_null",
    "primary.ranks.trailing_ratio_random_to_null",
    # the replication block, on the renamed keys
    "replication.2011-2012.n", "replication.2011-2012.is_primary",
    "replication.2011-2012.log_components_at_cutoff",
    "replication.2011-2012.levels_components_at_cutoff",
    "replication.2011-2012.degrees_of_freedom",
    "replication.2011-2012.cumulative_variance_cutoff",
    "replication.2011-2012.smallest_exact.features",
    "replication.2011-2012.routes.to_hgb.MCHC*HCT/100.mae",
]


@pytest.mark.parametrize("key", ABLATION_KEYS)
def test_ablation_json_carries_the_keys_the_figures_read(ablation_json, key):
    deref(ablation_json, key, "ablation.json")


@pytest.mark.parametrize("key", BENCHMARK_KEYS)
def test_benchmark_json_carries_the_keys_the_figures_read(benchmark_json, key):
    deref(benchmark_json, key, "benchmark.json")


@pytest.mark.parametrize("key", WEIGHTED_KEYS)
def test_weighted_json_carries_the_keys_the_figures_read(weighted_json, key):
    deref(weighted_json, key, "weighted.json")


@pytest.mark.parametrize("key", IDENTITY_KEYS)
def test_identity_audit_json_carries_the_keys_the_readme_quotes(identity_json, key):
    deref(identity_json, key, "identity_audit.json")


# The lists above run one direction: a key the paper reads must be present. This
# runs the other: a block the artefact publishes must be a block somebody claimed.
#
# The gap between the two is where the real risk was. Ten top-level blocks -
# ablation's contrast_verdict, surviving_contrasts, null_contrasts,
# out_of_cycle_ordering, majority_baseline_severity, contrasts_sign_convention,
# benchmark's selection_table, accuracy_spread_across_models and
# identity_gap_across_estimators, and weighted's provenance_note - appeared
# nowhere outside the script that wrote them. That is not dead output: several are
# the verdicts a reviewer will quote, and selection_table is the entire answer to
# "why Random Forest?". It only meant nothing would notice if one of them were
# renamed, emptied or dropped. Naming them here is what makes a deletion loud.
EXPECTED_TOP_LEVEL = {
    "ablation.json": {
        "cohort", "n_train", "n_val", "n_test", "fit_rule", "n_test_clusters",
        "majority_baseline_binary", "majority_baseline_severity",
        "pr_auc_no_skill", "variance_estimator", "estimand", "rungs",
        "contrasts", "contrasts_sign_convention", "variance_scales",
        "surviving_contrasts", "null_contrasts", "train_size_sensitivity",
        "contrast_verdict", "complete_case_sensitivity", "cluster_holdout",
        "tracking", "decomposition", "out_of_cycle", "out_of_cycle_ordering"},
    "benchmark.json": {
        "cohort", "n_train", "n_val", "n_test", "no_information_rate",
        "pr_auc_no_skill", "seed", "n_folds", "n_cv_repeats", "n_replicates",
        "selection_metric", "like_for_like_model", "variance_estimator",
        "estimand", "binary", "severity", "significance", "selected_model",
        "selection_table", "accuracy_spread_across_models",
        "identity_gap_across_estimators", "answers"},
    "weighted.json": {
        "cohort", "weight_variable", "weight_source_column", "seed", "n_folds",
        "n_replicates", "provenance_note", "design", "prevalence", "subgroups",
        "severity", "model_oof", "fit_weight_sensitivity"},
    "identity_audit.json": {
        "primary_cycle", "primary", "replication", "replication_note"},
    "importance.json": {"rung", "scoring", "n_repeats", "features"},
}


def test_no_result_file_publishes_or_drops_a_block_nobody_declared(
        ablation_json, benchmark_json, weighted_json, identity_json):
    """
    Set equality, not containment, and in both directions on purpose.

    A missing block is a stage that wrote less than it used to and still exited 0.
    An extra block is output the paper has no account of - and the manifest hashes
    the file as a whole, so an unexplained block is provenance a reader cannot
    check against anything. Either way the diff is printed by name.
    """
    import json as _json
    got = {"ablation.json": ablation_json, "benchmark.json": benchmark_json,
           "weighted.json": weighted_json, "identity_audit.json": identity_json,
           "importance.json": _json.loads(
               (ROOT / "results" / "importance.json").read_text(encoding="utf-8"))}
    for name, blob in got.items():
        diff = sorted(set(blob) ^ EXPECTED_TOP_LEVEL[name])
        assert not diff, f"{name}: undeclared or missing top-level blocks {diff}"


def test_the_blocks_nothing_outside_their_own_script_reads_are_not_empty(
        ablation_json, benchmark_json, weighted_json):
    """
    Presence is not enough for these ten, because nothing else would ever look.

    A key whose only reader is this test can be reduced to `{}`, `[]` or `""` by a
    refactor and stay present. contrast_verdict is a sentence a reviewer reads;
    selection_table is seven validation scores; out_of_cycle_ordering is the
    replication claim. Emptiness is the failure mode that presence-checking misses.
    """
    assert ablation_json["contrast_verdict"].startswith("5 of 6"), (
        "the pre-declared contrast tally moved: "
        f"{ablation_json['contrast_verdict'][:80]}")
    assert len(ablation_json["surviving_contrasts"]) == 5
    assert len(ablation_json["null_contrasts"]) == 1
    assert (len(ablation_json["surviving_contrasts"])
            + len(ablation_json["null_contrasts"])
            == len(ablation_json["contrasts"])), (
        "a contrast is neither surviving nor null, so one of the three lists is "
        "no longer a partition of the family")
    assert len(ablation_json["contrasts_sign_convention"]) > 100
    for pair, rec in ablation_json["out_of_cycle_ordering"].items():
        assert len(rec["gap_by_cycle"]) == 4, pair
        assert rec["smallest_margin_cycle"] in rec["gap_by_cycle"], pair
    assert len(benchmark_json["selection_table"]) == 7, (
        "the selection table is the answer to R6-2 and it is over the real "
        "estimators only, not the two Dummy baselines")
    assert benchmark_json["selected_model"] in benchmark_json["selection_table"]
    assert (benchmark_json["like_for_like_model"]
            in benchmark_json["selection_table"])
    assert len(weighted_json["provenance_note"]) > 100


def test_the_answer_to_why_random_forest_counts_the_models_it_searched(
        benchmark_json):
    """
    The count in the prose has to be the count in the table beside it.

    This string said "Nine estimators were run" and then quoted a search over
    seven: nine is the size of `models`, which includes the two Dummy baselines,
    and the baselines were never selection candidates. It is the answer to the
    reviewer question that asks how the model was chosen, so a nine there invites
    a reviewer to check nine rows against a seven-row table. The neighbouring
    spread_vs_rung had been deriving the same count correctly the whole time.
    """
    ans = benchmark_json["answers"]["why_random_forest"]
    n = len(benchmark_json["selection_table"])
    assert f"{n} estimators" in ans, ans
    assert "Nine estimators" not in ans, ans
    # And both named models have to appear with the scores the table holds.
    for name in (benchmark_json["selected_model"],
                 benchmark_json["like_for_like_model"]):
        assert f"{benchmark_json['selection_table'][name]:.4f}" in ans, name
    spread = benchmark_json["answers"]["spread_vs_rung"]
    assert f"of {benchmark_json['identity_gap_across_estimators']['n_estimators']}" \
        in spread, spread


def test_the_identity_component_counts_are_not_called_degrees_of_freedom(
        identity_json):
    """
    A PCA truncation and an algebraic rank are two different numbers here.

    The block has six published variables and three exact identities, so it has
    three degrees of freedom, and that is arithmetic - it does not depend on any
    variance cutoff. The log spectrum happens to agree (3 components at 99.9%)
    and the levels spectrum does not (4), which is the whole point of reporting
    both. While the replication block called those counts `log_dof` and
    `levels_dof`, the file invited a reader to quote the levels number as the
    degrees of freedom and get 4 - contradicting the same file's own
    `degrees_of_freedom` field two keys away.
    """
    for cycle, rec in identity_json["replication"].items():
        assert "log_dof" not in rec and "levels_dof" not in rec, (
            f"{cycle}: a component count is published as a degrees-of-freedom key")
        assert rec["degrees_of_freedom"] == 3, cycle
        assert rec["log_components_at_cutoff"] == 3, cycle
        assert rec["levels_components_at_cutoff"] == 4, cycle
    ranks = identity_json["primary"]["ranks"]
    assert "log_rank_at_cutoff" not in ranks, (
        "log_rank_at_cutoff is back; it is a third copy of ranks.logs.k")
    assert ranks["degrees_of_freedom"] == 3
    assert ranks["n_identities"] == 3
    assert ranks["levels_minus_logs"] == (
        ranks["levels"]["k"] - ranks["logs"]["k"])


def test_the_replication_block_covers_every_cycle_including_the_primary(
        identity_json, dp):
    """
    The identity audit replicates on all four cycles, unlike the ablation.

    ablation.json's out_of_cycle deliberately excludes the primary, because a
    model cannot be a held-out replication of the rows it was fitted on. This
    file has no model in it: the identities are arithmetic, so including the
    primary cycle is what lets a reader see that the audit's own numbers for
    2017-2020 match the `primary` block computed separately above.
    """
    got = set(identity_json["replication"])
    assert got == set(dp.CYCLES), got
    assert identity_json["primary_cycle"] == dp.PRIMARY
    prim = identity_json["replication"][dp.PRIMARY]
    assert prim["is_primary"] is True
    assert prim["n"] == identity_json["primary"]["n"]
    for cycle, rec in identity_json["replication"].items():
        assert rec["is_primary"] is (cycle == dp.PRIMARY), cycle


def test_every_hgb_route_recovers_the_label_it_was_supposed_to_hide(identity_json):
    """
    The claim the paper rests on, checked as a number rather than as prose.

    Each of the four arithmetic routes to hemoglobin has to reconstruct it to
    well inside the 0.1 g/dL grid it is published on, and has to reproduce the
    WHO binary label at a rate a reviewer would call recovery. If a future change
    to dataprep.HGB_ROUTES ever made one of these routes stop working, the
    leakage argument would weaken and nothing else here would notice.
    """
    routes = identity_json["primary"]["routes"]["to_hgb"]
    assert len(routes) == 4, sorted(routes)
    for name, rec in routes.items():
        assert rec["mae"] < 0.05, f"{name}: mae {rec['mae']} is above half a grid step"
        assert rec["label_agreement_who"] > 0.99, name
        assert rec["needs"] and len(rec["needs"]) == len(rec["needs_columns"]), name


def test_the_renamed_key_is_gone_and_not_merely_shadowed(ablation_json):
    """
    `external` became `out_of_cycle`. Both present would be the worse outcome.

    A rename that leaves the old key behind is how two figures end up reading two
    different numbers under one name, and neither of them fails.
    """
    assert "out_of_cycle" in ablation_json
    assert "external" not in ablation_json, "the old key survived the rename"


def test_the_out_of_cycle_block_holds_only_the_three_replication_cycles(ablation_json, dp):
    """
    2017-2020 is the training cycle, so it cannot be one of its own replications.

    fig10 draws the primary held-out block as a separate reference bar built from
    `rungs`, which is right; what would be wrong is out_of_cycle containing a
    2017-2020 entry, because then the same rows would appear as both the fit and a
    replication of it.
    """
    got = set(ablation_json["out_of_cycle"])
    assert got == {"2011-2012", "2013-2014", "2015-2016"}, got
    assert dp.PRIMARY not in got
    assert got == set(dp.CYCLES) - {dp.PRIMARY}


def test_the_shipped_models_score_exactly_what_the_ablation_reports(
        ablation_json, benchmark_json):
    """
    Three files publish the same two accuracies. They have to be the same number.

    anemia_app.py trains the models that ship in models/*.pkl and writes
    model_info.json; ablation.py's L6 rung and benchmark.py's rf@L6 are the same
    estimator on the same rung, the same split and the same seed. So a reader can
    line up model_info.json against the paper's headline table, and the app's
    own docstring promises exactly that comparison.

    It has already been broken once, in the direction that is hardest to see: the
    severity forest in the app used 400 trees while every pipeline script used
    300, so the number the app existed to publish was 0.9502 against the paper's
    0.9515 - two accuracies for one model, neither wrong, and nothing anywhere
    that said why. dp.N_TREES is now the single source, and this is the check that
    it stays that way. Exact equality, not approximate: these are the same fit,
    so any difference at all is a configuration that drifted.
    """
    import json
    mi = json.loads((ROOT / "models" / "model_info.json").read_text("utf-8"))
    L = "L6_identity_free_demo"
    for head in ("binary", "severity"):
        want = mi[head]["test_accuracy"]
        assert ablation_json["rungs"][L][head]["test_accuracy"] == want, (
            f"{head}: model_info {want} vs ablation "
            f"{ablation_json['rungs'][L][head]['test_accuracy']}")
        assert benchmark_json[head][L]["models"]["rf"]["test_accuracy"] == want, (
            f"{head}: model_info {want} vs benchmark rf "
            f"{benchmark_json[head][L]['models']['rf']['test_accuracy']}")
    assert (ablation_json["rungs"][L]["binary"]["cv_accuracy_mean"]
            == mi["binary"]["cv_accuracy_mean"])


def test_the_two_files_that_publish_one_design_se_publish_one_number(
        ablation_json, benchmark_json):
    """
    The discrepancy dp.N_REP exists to close, pinned so it cannot reopen quietly.

    ablation.py and benchmark.py both estimate the design-based SE of rf@L6's test
    accuracy - same 2,432 rows, same 49 PSU cells, same 25 residual df, same seed.
    They published 0.00407619 and 0.00408166. Neither was wrong: the replicate
    count was 1000 in one file and 500 in the other, and the gap was Monte-Carlo
    noise in the spread. But a reader who lines the two files up finds two numbers
    for one quantity with nothing in either file to explain it.

    Exact equality is the right assertion precisely because it is fragile. The two
    scripts share dp.N_REP, dp.SEED and dp.rao_wu_weights; if any one of those
    stops being shared, this fails, which is the only warning a reader would ever
    get before the two numbers diverge again by an amount too small to notice.
    """
    L = "L6_identity_free_demo"
    for head in ("binary", "severity"):
        a = ablation_json["rungs"][L][head]["test_accuracy_design_se"]
        b = benchmark_json[head][L]["models"]["rf"]["test_accuracy_design_se"]
        assert a == b, f"{head}: ablation {a!r} vs benchmark {b!r}"
    assert ablation_json["rungs"][L]["binary"]["test_accuracy_design_se"] > 0


def test_the_weight_cv_is_one_number_and_reproduces_the_kish_factor(weighted_json):
    """
    Same shape of defect on the other side: one CV, published twice, two values.

    svy_prop derived it from the Kish factor and the design block computed it
    directly with ddof=1, so weighted.json carried 1.1904091 and 1.1903602 for the
    same weights on the same rows - a gap of exactly sqrt(n/(n-1)).

    The identity is the reason ddof=0 had to win, so the identity is what is
    checked: the file PRINTS "the Kish factor 1 + CV^2" as the explanation of
    deff_unequal_weighting, and with ddof=1 that printed sentence did not
    reproduce the number printed beside it. Every block that carries both keys is
    checked, not just the design block, because the duplication was across blocks.
    """
    design_cv = weighted_json["design"]["weight_cv"]
    seen = 0
    for block in ("prevalence", "subgroups", "severity"):
        for name, rec in weighted_json[block].items():
            if "weight_cv" not in rec:
                continue
            cv, deff = rec["weight_cv"], rec["deff_unequal_weighting"]
            assert cv is not None, f"{block}/{name}"
            assert abs((1.0 + cv * cv) - deff) < 1e-12, (
                f"{block}/{name}: 1 + CV^2 = {1 + cv * cv!r} but "
                f"deff_unequal_weighting = {deff!r}")
            seen += 1
    assert seen >= 15, f"only {seen} blocks carried a weight_cv"
    # The whole-cohort rows are the ones that were inconsistent with each other.
    whole = [weighted_json["prevalence"]["Anemia"]["weight_cv"],
             weighted_json["subgroups"]["all"]["weight_cv"],
             weighted_json["severity"]["Normal"]["weight_cv"]]
    assert whole == [design_cv] * 3, (design_cv, whole)


def test_the_cluster_optimism_disclosure_is_read_from_ablation_not_retyped(
        ablation_json, weighted_json):
    """
    The honesty disclosure had three hand-copied numbers in it, and one was wrong.

    weighted.model_oof.shared_cluster_optimism quotes the magnitude of the
    row-wise-fold optimism, and its source is ablation.cluster_holdout in another
    file. It used to carry 0.002, 0.006 and "roughly 0.4x to 1.1x" as literals;
    the true ratio at the low end is 0.32x. A stale number inside the paragraph
    that exists to admit a limitation is worse than the limitation.
    """
    ch = ablation_json["cluster_holdout"]
    d = weighted_json["model_oof"]["shared_cluster_optimism"]
    assert d["available"] is True, d.get("reason")
    assert d["accuracy_points_mean"] == ch["absolute_optimism_mean"]
    assert d["accuracy_points_worst_case"] == ch["absolute_optimism_max"]
    lo, hi = weighted_json["model_oof"]["metrics"]["accuracy"]["ci95_wald"]
    half = (hi - lo) / 2.0
    assert d["exceeds_ci_half_width"] is bool(ch["absolute_optimism_max"] > half)
    assert f"{ch['absolute_optimism_mean'] / half:.2f}x" in \
        d["relative_to_ci_half_width"]
    assert f"{ch['absolute_optimism_max'] / half:.2f}x" in \
        d["relative_to_ci_half_width"]


def test_no_rung_optimism_is_compared_against_a_zero_width_interval(ablation_json):
    """
    L7's design interval has zero width, so "exceeds its own half-width" is empty.

    Hemoglobin plus demographics reproduces the WHO label on every replicate, so
    L7_hgb_demo's test-accuracy interval is a point. Without the `> 0` guard any
    positive optimism at all clears it, and the rung with no measurable variance
    would be reported as the one whose optimism is worrying. The ratio is recorded
    as null there rather than as a large number.
    """
    ch = ablation_json["cluster_holdout"]
    for rung, ratio in ch["optimism_vs_own_ci_half_width"].items():
        ci = ablation_json["rungs"][rung]["binary"]["test_accuracy_ci95"]
        half = (ci[1] - ci[0]) / 2.0
        if half == 0.0:
            assert ratio is None, f"{rung}: ratio {ratio} against a point interval"
            assert rung not in ch["rungs_where_optimism_exceeds_own_ci"], rung
        else:
            assert ratio is not None, rung
    assert "L7_hgb_demo" in ch["optimism_vs_own_ci_half_width"]


def test_every_figure_the_pipeline_declares_exists_in_both_formats(manifest_json):
    """PNG for the draft, PDF for the camera-ready. A figure is both or neither."""
    figs = {Path(k).stem for k in manifest_json["artefacts"]
            if k.startswith("figures/")}
    assert len(figs) == 10, sorted(figs)
    for stem in figs:
        for ext in ("png", "pdf"):
            p = ROOT / "figures" / f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size > 0, f"{stem}.{ext}"
    assert "fig10_out_of_cycle" in figs
    assert not any("external" in f for f in figs), "a fig*_external file survived"


def test_every_results_json_is_parseable_and_finite(manifest_json):
    """
    Valid JSON is not enough: NaN and Infinity are things json.load accepts.

    Every script here dumps with allow_nan=False for exactly this reason, and this
    is the check on the other side of that. A bare NaN in results/ is what makes a
    figure draw an empty axis and a table print "nan" in a submitted paper.
    """
    for p in sorted((ROOT / "results").glob("*.json")):
        text = p.read_text(encoding="utf-8")
        blob = json.loads(text)
        assert isinstance(blob, dict), p.name
        for token in ("NaN", "Infinity", "-Infinity"):
            assert f": {token}" not in text and f", {token}" not in text, \
                f"{p.name} contains a bare {token}"
