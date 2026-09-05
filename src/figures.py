"""
Step 7 - every figure the paper needs, regenerated from results/*.json.

Nothing here invents a number. The panels read ablation.json, benchmark.json
and weighted.json, and only refit a model when a figure genuinely needs
something the JSON cannot hold (per-threshold PR curves, permutation
importances, confusion matrices).

Figures produced, in FIGURES/ as both PNG (300 dpi) and PDF (vector, for the
IEEE template):

  fig1_flowchart      methodology, end to end                    -> R6-3
  fig2_ladder         Hb recoverability vs accuracy, per rung     the main result
  fig3_decomposition  the 2x2: identity leakage x label definition
  fig4_pr_curves      PR for the 7 real estimators at L6, with operating
                      points                                        -> R4-2, R6-4
  fig5_calibration    reliability diagram and Brier scores        -> R4-5
  fig6_confusion      L1 vs L6, binary and severity
  fig7_importance     MDI vs permutation importance, and why they disagree
  fig8_prevalence     unweighted vs survey-weighted, by subgroup  -> R4 survey
  fig9_contrasts      the six pre-declared contrasts and the three
                      variance scales they can be measured on
  fig10_out_of_cycle  out-of-cycle replication, 2011-12 to 2017-20

Two of these exist because a claim in the paper is otherwise unfalsifiable.
fig9 shows which contrasts survive Holm under a design-based variance AND how
much the variance scale itself matters, so a reader can see that the naive
cluster bootstrap is narrower rather than being told. fig10 shows the ladder
reproducing on three cycles the models never saw.

Run:  python src/figures.py            (after ablation, benchmark and weighted)
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve

import benchmark as bm
import dataprep as dp
from paths import FIGURES, RESULTS

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

COL1, COL2 = 3.5, 7.16          # IEEE single and double column width, inches

HEADLINE = "L6_identity_free_demo"
SUBMITTED = "L1_paper_leaky"
LIKE_FOR_LIKE = "rf"            # the estimator the manuscript used, so every
                                # leakage comparison is like-for-like

# Permutation repeats for fig7. One constant because the number appeared three
# times - the permutation_importance call, the panel title "(30 repeats)", and
# the n_repeats field written into importance.json - so changing it in the call
# alone would have published a figure and an artefact that both misreported how
# the numbers beside them were produced.
PERM_REPEATS = 30

RUNG_SHORT = {
    "L0_hgb_direct": "L0\nHGB given",
    "L1_paper_leaky": "L1\nRBC MCV MCH\nsubmitted",
    "L1_paper_leaky_demo": "L1d\nL1+demo",
    # Spelled out rather than "+MCHC HCT". Every other "+" label here means "the
    # rung above plus these", and that reading is false for L2: it adds MCHC and
    # HCT but also DROPS MCH, so a reader following the convention would have
    # inferred a five-feature rung that does not exist.
    "L2_mchc_hct": "L2\nRBC MCV\nMCHC HCT",
    "L3_hct_only": "L3\nRBC MCV\nHCT",
    "L4_identity_free": "L4\nRBC MCV RDW",
    "L5_identity_free_plus": "L5\n+PLT WBC",
    "L6_identity_free_demo": "L6\n+age sex\n+preg\ndefensible",
    "L7_hgb_demo": "L7\nHGB+demo",
}
# These are display labels for dp.LADDER's rungs, hand-written because the
# feature-name abbreviations and the line breaks are typographic choices no
# generator would make well. The one thing that must not drift is the key set:
# a rung added to the ladder with no label here would raise a KeyError deep
# inside a figure, and a label left behind after a rung is renamed would sit
# unused and unnoticed. Checked at import, so both fail immediately and by name.
assert set(RUNG_SHORT) == set(dp.LADDER), (
    "RUNG_SHORT and dp.LADDER disagree: "
    f"labels with no rung {sorted(set(RUNG_SHORT) - set(dp.LADDER))}, "
    f"rungs with no label {sorted(set(dp.LADDER) - set(RUNG_SHORT))}")
MODEL_LABEL = {
    "majority": "majority baseline", "stratified": "stratified baseline",
    "logreg": "logistic regression", "svm_rbf": "SVM (RBF)",
    "dtree": "decision tree", "rf": "random forest",
    "extratrees": "extra trees", "histgb": "hist. gradient boosting",
    "xgboost": "XGBoost",
}
FEAT_LABEL = {
    dp.RBC: "RBC", dp.MCV: "MCV", dp.MCH: "MCH", dp.MCHC: "MCHC",
    dp.HCT: "HCT", dp.RDW: "RDW", dp.PLT: "platelets", dp.WBC: "WBC",
    dp.HGB: "hemoglobin", dp.AGE: "age", dp.SEX: "sex",
    dp.PREG: "pregnancy",
}


def _n_baselines(bench):
    """
    How many of the benchmarked estimators are trivial baselines.

    Counted, not written down. fig1's caption said "2 of them trivial baselines"
    as a literal while the estimator total right next to it was computed from the
    artefact, so adding or dropping a Dummy would have produced a figure that
    contradicted itself. bm.BASELINES is the single declaration.
    """
    return sum(1 for m in bench["binary"][HEADLINE]["models"] if m in bm.BASELINES)


def load(name):
    with open(RESULTS / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    for ext in ("png", "pdf"):
        # metadata CreationDate=None: matplotlib otherwise stamps the wall clock
        # into every PDF, so all ten re-hashed on every run and run_manifest.json
        # could never tell an inert change from a real one.
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300,
                    metadata=({"CreationDate": None} if ext == "pdf" else None))
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


def box(ax, x, y, w, h, text, fc="#eaf0f7", ec="#3b5b7d", fs=7.4, weight=None):
    """One flowchart node. pad is deliberately tiny so h is the real height."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, linespacing=1.35)


def arrow(ax, xy_from, xy_to, style="-|>", color="#3b5b7d", ls="-"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=9, color=color,
                                 linewidth=1.0, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def fig1_flowchart(ab, wt, bench):
    """R6-3: the methodology diagram the reviewer asked for, with real counts."""
    c, d = ab["cohort"], wt["design"]
    pw = wt["prevalence"]["Anemia"]
    # ab["out_of_cycle"] is required, not optional. `.get(..., {})` let a run that
    # never reached the replication step still draw a flowchart claiming the
    # replication happened - the counts below would silently read 0 cycles and the
    # figure would look finished. ablation.py always writes the key, so a KeyError
    # here means the artefact is truncated and the figure should not be produced.
    ext = ab["out_of_cycle"]
    fig, ax = plt.subplots(figsize=(COL2, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    W, H = 0.46, 0.068
    ax.text(0.5, 0.988, "Anemia detection pipeline, NHANES 2017-March 2020",
            ha="center", fontsize=10, fontweight="bold")

    box(ax, 0.28, 0.936, 0.30, 0.056,
        f"P_CBC.XPT\n{c['cbc_records']:,} records")
    box(ax, 0.72, 0.936, 0.30, 0.056,
        f"P_DEMO.XPT\n{c['demo_records']:,} records")
    box(ax, 0.50, 0.858, W, 0.048,
        f"inner merge on SEQN  ->  {c['after_demo_merge']:,}")
    box(ax, 0.50, 0.775, W, H,
        f"complete case on the {len(dp.ALL_FEATURE_COLS)} CBC variables\n"
        f"{c['complete_on_all_features']:,} retained "
        f"({c['excluded_pct']}% excluded, not MCAR)")
    # The cut-offs come out of the cohort audit. They used to be the literal
    # string "11.0  11.5  12.0  12.0  13.0", re-typed from dataprep.who_thresholds
    # into a figure nothing cross-checks - and it listed 12.0 twice, which is true
    # of the WHO bands and false of the set of distinct cut-offs actually applied.
    thr = c["who_thresholds"]
    box(ax, 0.50, 0.683, W, H,
        "WHO age / sex / pregnancy cut-offs\n"
        + "  ".join(f"{v:.1f}" for v in thr["thresholds_gdl"]) + " g/dL")
    box(ax, 0.50, 0.591, W, H,
        f"labels: binary anemia {pw['prop_unweighted']*100:.2f}% unweighted\n"
        f"4-class WHO severity, merged to 3")
    box(ax, 0.50, 0.502, W, 0.056,
        f"ONE stratified 60/20/20 split\n"
        f"train {ab['n_train']:,} / val {ab['n_val']:,} / test {ab['n_test']:,}",
        fc="#fdf3e3", ec="#b07d2a")

    box(ax, 0.255, 0.392, 0.40, 0.082,
        f"leakage ladder, {len(ab['rungs'])} rungs L0-L7\n"
        "Hb recoverability, best of\n{exact, RF, OLS, OLS in logs}",
        fc="#f2e9f5", ec="#7a4f8a")
    box(ax, 0.745, 0.392, 0.40, 0.082,
        f"{len(bench['binary'][HEADLINE]['models']) - _n_baselines(bench)} "
        f"estimators + {_n_baselines(bench)} trivial baselines\n"
        f"at L1 / L4 / L6\n"
        f"selected on validation {bench['selection_metric']}",
        fc="#e8f3ec", ec="#3f7d55")

    box(ax, 0.50, 0.272, 0.76, H,
        f"evaluation: {bench['n_folds']}x{bench['n_cv_repeats']} repeated CV, "
        f"Rao-Wu replicate weights ({ab['n_test_clusters']} PSUs, "
        f"{wt['design']['df']} df),\n"
        f"{len(ab['contrasts'])} pre-declared contrasts, Holm-corrected; "
        f"McNemar and the i.i.d. row bootstrap shown alongside")
    box(ax, 0.50, 0.174, 0.76, H,
        f"survey design: {d['n_strata']} strata, {d['total_psus']} PSUs, "
        f"{wt['weight_source_column']}\nweighted prevalence "
        f"{pw['prop_weighted']*100:.2f}%  "
        f"(design effect {pw['design_effect']:.2f}, effective n "
        f"{pw['n_effective_design']:,.0f})")
    box(ax, 0.50, 0.082, 0.76, 0.056,
        f"out-of-cycle replication on {len(ext)} earlier cycles "
        f"({', '.join(ext)})\nfitted on 2017-2020 only, never refitted",
        fc="#eef4f8", ec="#3b5b7d")
    box(ax, 0.50, 0.028, 0.56, 0.040,
        "screening tool: L6 features, hemoglobin never an input",
        fc="#fdecec", ec="#a63a3a", weight="bold")

    # main spine
    arrow(ax, (0.28, 0.906), (0.40, 0.884))
    arrow(ax, (0.72, 0.906), (0.60, 0.884))
    for y0, y1 in [(0.832, 0.812), (0.740, 0.720), (0.648, 0.628),
                   (0.556, 0.532), (0.236, 0.210), (0.138, 0.112),
                   (0.052, 0.048)]:
        arrow(ax, (0.50, y0), (0.50, y1))
    arrow(ax, (0.44, 0.472), (0.31, 0.436))
    arrow(ax, (0.56, 0.472), (0.69, 0.436))
    arrow(ax, (0.29, 0.348), (0.42, 0.310))
    arrow(ax, (0.71, 0.348), (0.58, 0.310))

    # hemoglobin: the whole point of the study is where this arrow does NOT go
    box(ax, 0.128, 0.650, 0.235, 0.110,
        "LBXHGB\nhemoglobin\n\nderives the labels\nand nothing else",
        fc="#fdecec", ec="#a63a3a", fs=7.2)
    arrow(ax, (0.248, 0.678), (0.272, 0.686), color="#a63a3a", ls="--")
    arrow(ax, (0.248, 0.620), (0.272, 0.594), color="#a63a3a", ls="--")
    ax.text(0.128, 0.582,
            "present as a feature only in\nL0 and L7, which exist to\n"
            "measure the leak, not to\npredict anything. MCH x RBC\n"
            "/ 10 puts hemoglobin back,\nwhich is what the ladder\nmeasures.",
            ha="center", va="top", fontsize=6.5, color="#a63a3a",
            linespacing=1.35)
    save(fig, "fig1_flowchart")


def fig2_ladder(ab):
    """
    The paper's central figure: classification accuracy tracks hemoglobin
    recoverability, not diagnostic signal.

    Two stacked panels sharing the rung axis, not one axes with a twin. The
    docstring used to describe a "left axis / right axis" pair, which was the
    first version's layout and stopped being true when accuracy and MAE were
    split apart; a reader following it would have looked for a second y-axis on
    the accuracy panel that is not there.

    Upper panel - test accuracy with its design-based CI, and PR-AUC, against
                  the no-information rate and the PR-AUC no-skill line.
    Lower panel - MAE of the best hemoglobin reconstruction available from that
                  rung, on an INVERTED axis so 'easy to recover' points the same
                  way as 'high accuracy' above it, plus WHO label agreement on a
                  right-hand twin. The two panels tracking each other down the
                  ladder is the claim being made.

    Best-of-family rather than one regressor: an identity route when the rung
    admits one, otherwise the better of random forest, OLS and OLS in logs. A
    single learner would understate recoverability wherever it happens to be
    the wrong functional form, and the identities are products, so in logs they
    are linear and a log fit wins on the identity-free rungs.

    "Best" here is ablation's `selected` field: the family member chosen by
    VALIDATION MAE and then scored on test, so this axis contains no test-set
    selection. ablation.json also carries `best_on_test`, the oracle that
    selecting on test would have produced; it is deliberately not what is
    plotted.
    """
    rungs = list(ab["rungs"])
    acc = [ab["rungs"][r]["binary"]["test_accuracy"] for r in rungs]
    lo = [ab["rungs"][r]["binary"]["test_accuracy_ci95"][0] for r in rungs]
    hi = [ab["rungs"][r]["binary"]["test_accuracy_ci95"][1] for r in rungs]
    pr = [ab["rungs"][r]["binary"]["pr_auc"] for r in rungs]
    best = [ab["rungs"][r]["recoverability"]["selected"] for r in rungs]
    mae = [b["mae_gdl"] for b in best]
    agree = [b["label_agreement"] for b in best]
    maj = ab["majority_baseline_binary"]
    noskill = ab["pr_auc_no_skill"]
    x = np.arange(len(rungs))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL2, 6.2), sharex=True,
                                  height_ratios=[1.25, 1])

    ax.errorbar(x, acc, yerr=[np.array(acc) - lo, np.array(hi) - np.array(acc)],
                marker="o", ms=5, lw=1.6, capsize=3, color="#c44e52",
                label="test accuracy (design-based 95% CI)")
    ax.plot(x, pr, marker="s", ms=4.5, lw=1.6, color="#4c72b0",
            label="PR-AUC")
    ax.axhline(maj, ls="--", lw=1.1, color="#666",
               label=f"no-information rate {maj:.4f}")
    ax.axhline(noskill, ls=":", lw=1.1, color="#4c72b0",
               label=f"PR-AUC no-skill = prevalence {noskill:.4f}")
    ax.set_ylabel("classification performance")
    ax.set_ylim(0.0, 1.06)
    # Both legends sit in the empty mid-band rather than a corner: the bottom
    # right holds the PR-AUC no-skill line (drawn just above at `noskill`, a low
    # single-digit-percent prevalence - the value used to be re-typed here as
    # "0.0958", which is a number that goes stale the first time the cohort or
    # the label definition changes and that nothing cross-checks), and the
    # bottom left of the lower panel holds the L3-L5 estimator labels.
    ax.legend(loc="center", framealpha=0.95, ncol=2)
    ax.set_title("Anemia classification performance by feature set")
    for i, r in enumerate(rungs):
        if r in (SUBMITTED, "L7_hgb_demo", HEADLINE):
            ax.annotate(f"{acc[i]:.4f}", (x[i], hi[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        fontweight="bold")

    ax2.plot(x, mae, marker="D", ms=4.5, lw=1.6, color="#55a868",
             label="hemoglobin MAE, g/dL (chosen on validation)")
    ax2.set_ylabel("Hb reconstruction error (g/dL)")
    ax2.invert_yaxis()
    ax3 = ax2.twinx()
    ax3.plot(x, np.array(agree) * 100, marker="^", ms=4.5, lw=1.4, ls=":",
             color="#8172b2", label="WHO label agreement (%)")
    ax3.set_ylabel("label agreement (%)")
    ax3.grid(False)
    ax3.spines["right"].set_visible(True)
    pad = max(0.4, (100.0 - min(agree) * 100) * 0.18)
    ax3.set_ylim(min(agree) * 100 - pad, 100.0 + pad)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center", framealpha=0.95)
    ax2.set_title("Hemoglobin recoverability from the same feature set "
                  "(axis inverted: higher = more leakage)", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([RUNG_SHORT[r] for r in rungs], fontsize=5.8)
    ax2.tick_params(axis="x", pad=2)
    ax2.margins(y=0.16)
    worst = int(np.argmax(mae))
    # Keep the selected recoverability route visible without allowing adjacent
    # long algebraic labels to run into one another at the crowded middle
    # rungs. The full route names remain in ablation.json; these are compact
    # display labels only.
    route_labels = {
        "HGB present verbatim": "HGB verbatim",
        "MCH*RBC/10": "MCH x RBC/10",
        "MCHC*HCT/100": "MCHC x HCT/100",
        "MCHC*MCV*RBC/1000": "MCHC x MCV x RBC/1000",
        "MCH*HCT/MCV": "MCH x HCT/MCV",
        "ols": "OLS",
        "ols in logs": "OLS (log)",
        "random forest": "RF",
    }
    for i, b in enumerate(best):
        # The MAE axis is inverted, so a positive offset moves the label toward
        # zero error. Labels sit below their marker, except at the worst rung,
        # which is already against the bottom of the axis.
        dy = 14 if i == worst else (14 if i % 2 == 0 else -13)
        ha = "left" if i == 0 else ("right" if i == len(best) - 1 else "center")
        label = route_labels.get(b["estimator"], b["estimator"])
        ax2.annotate(label, (x[i], mae[i]), textcoords="offset points",
                     xytext=(0, dy), ha=ha, fontsize=5.2, color="#2f6b3f")
    save(fig, "fig2_ladder")


def fig3_decomposition(ab):
    """
    The submitted number split into its two independent causes, as a 2x2.

    Left panel: raw accuracy and PR-AUC with each cell's OWN reference line,
    because changing the label changes the class balance and therefore changes
    what "no information" means. Right panel: the same four cells on the LIFT
    scale (value minus that cell's own floor), which is the only scale on which
    the two effects can be added, plus the interaction.

    Comparing the leaky/old-label cell with the clean/WHO-label cell directly
    moves both factors at once, so the factorial is not decoration: it is what
    stops the two effects being reported as one.
    """
    dec = ab["decomposition"]
    cells = {(c["features"], c["label"]): c for c in dec["cells"]}
    eff = dec["effects_on_lift_scale"]
    labels = sorted({c["label"] for c in dec["cells"]},
                    key=lambda s: 0 if "12" in s else 1)
    feats = [SUBMITTED, "L4_identity_free"]
    fname = {SUBMITTED: "RBC MCV MCH\n(identity present)",
             "L4_identity_free": "RBC MCV RDW\n(identity free)"}

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 0.95],
                                          "wspace": 0.52})
    for ax, metric, ttl in [(axes[0], "accuracy", "Accuracy"),
                            (axes[1], "pr_auc", "PR-AUC")]:
        w = 0.36
        xs = np.arange(len(labels))
        for j, f in enumerate(feats):
            vals = [cells[(f, lb)][metric] for lb in labels]
            bars = ax.bar(xs + (j - 0.5) * w, vals, w, label=fname[f],
                          color=["#c44e52", "#4c72b0"][j], edgecolor="white")
            ax.bar_label(bars, fmt="%.4f", fontsize=5.9, padding=2)
        ticks = []
        for i, lb in enumerate(labels):
            # The reference line has to match the metric. For accuracy that is
            # the no-information rate; for PR-AUC it is the prevalence of the
            # positive class, which is what a random ranker achieves. Its value
            # goes in the tick label rather than next to the line, because at
            # these bar heights there is no clear space beside the line.
            c = cells[(feats[0], lb)]
            ref = (c["no_information_rate"] if metric == "accuracy"
                   else c["pr_auc_no_skill"])
            ax.plot([i - w, i + w], [ref, ref], ls="--", lw=1.2, color="#333",
                    label="that cell's own floor" if i == 0 else None)
            tag = "NIR" if metric == "accuracy" else "no-skill"
            two_line = lb.replace(" ", "\n")
            ticks.append(f"{two_line}\n{tag} {ref:.4f}")
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=6.5)
        ax.set_title(ttl)
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("test-set value")
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(h0, l0, loc="upper center", bbox_to_anchor=(1.28, -0.20),
                   ncol=3, fontsize=6.6, frameon=False)

    # ---- the lift scale, where the two effects are additive ------------------
    ax = axes[2]
    # Colour travels with the key, not with the position. `cols` used to be a
    # separate list sliced `cols[:len(keys)]`, so the moment any one of these six
    # keys was absent from `eff` the surviving bars kept the FIRST n colours
    # rather than their own - a label fix would have been drawn in the identity
    # colour, silently, in the one figure whose entire job is telling the two
    # factors apart.
    order = [("identity_removal_under_old_label",
              "identity removal\nold label", "#c44e52"),
             ("identity_removal_under_who_label",
              "identity removal\nWHO label", "#c44e52"),
             ("label_correction_with_identity",
              "label fix\nidentity present", "#4c72b0"),
             ("label_correction_without_identity",
              "label fix\nidentity gone", "#4c72b0"),
             ("interaction", "interaction", "#8172b2"),
             ("total_leaky_old_vs_clean_who", "both, total", "#333333")]
    present = [(k, n, c_) for k, n, c_ in order if k in eff]
    keys = [k for k, _, _ in present]
    vals = [eff[k] for k in keys]
    names = [n for _, n, _ in present]
    cols = [c_ for _, _, c_ in present]
    yy = np.arange(len(keys))
    bars = ax.barh(yy, vals, color=cols)
    ax.bar_label(bars, fmt="%+.2f", fontsize=6.6, padding=2)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("lift points over each\ncell's own floor", fontsize=7.2)
    # The left edge is min(0, ...), not 0. The interaction is a difference of
    # differences and has no sign guarantee - it is +1.81 here - and a negative
    # bar drawn on an axis starting at 0 is clipped to nothing, so the figure
    # would show an absent effect rather than an opposing one.
    ax.set_xlim(min(0.0, min(vals) * 1.34), max(0.0, max(vals) * 1.34))
    ax.set_title("Effects on the lift scale", fontsize=8.5)

    base = cells[(SUBMITTED, labels[0])]
    clean = cells[("L4_identity_free", labels[1])]

    # Each main effect is measured at the OTHER factor's leaky level, so
    # 3.29 + 5.30 is not the total - it counts the 1.81 interaction twice, and
    # lands on 8.59, which matches neither the 9.91 it claims to decompose nor
    # the 6.78 that actually vanishes. Only the two sequential paths are
    # additive, and they disagree; that disagreement IS the interaction. The
    # title now states both paths instead of a sum no reader can reconcile.
    seq_a = (eff["identity_removal_under_old_label"]
             + eff["label_correction_without_identity"])
    seq_b = (eff["label_correction_with_identity"]
             + eff["identity_removal_under_who_label"])
    tot = eff["total_leaky_old_vs_clean_who"]
    assert abs(seq_a - tot) < 0.02 and abs(seq_b - tot) < 0.02, \
        f"lift-scale effects no longer decompose: {seq_a} {seq_b} vs {tot}"
    fig.suptitle(
        f"Decomposing the submitted {base['accuracy']*100:.2f}%: "
        f"{base['lift_points']:.2f} lift points, of which {tot:.2f} vanish when "
        f"both are fixed, leaving {clean['lift_points']:.2f}\n"
        f"that {tot:.2f} is "
        f"{eff['identity_removal_under_old_label']:.2f}+"
        f"{eff['label_correction_without_identity']:.2f} (identity first) or "
        f"{eff['label_correction_with_identity']:.2f}+"
        f"{eff['identity_removal_under_who_label']:.2f} (label first); "
        f"interaction {eff['interaction']:+.2f}, so neither order is canonical",
        fontsize=8.3, y=1.06)
    save(fig, "fig3_decomposition")


def refit(df, tr, te, rung, target="Anemia"):
    """
    Refit the benchmark's estimators on one rung and keep their outputs.

    benchmark.py is imported rather than copied so the model definitions cannot
    drift between the table and the figure.
    """
    X = df[dp.LADDER[rung]].to_numpy()
    y = df[target].to_numpy()
    fitted = {}
    # No scale_pos_weight argument: benchmark.models() sets class weighting
    # inside each estimator, so the figure and the table use one definition.
    for name, est in bm.models().items():
        est.fit(X[tr], y[tr])
        fitted[name] = {
            "est": est,
            "pred": est.predict(X[te]),
            "proba": est.predict_proba(X[te])[:, 1],
        }
    return X, y, fitted


def fig4_pr_curves(df, tr, te, fitted, bench):
    """
    R4-2 / R6-4: precision-recall for every comparator on the same rows.

    The four marked points are the thresholds benchmark.py chose on the
    VALIDATION split and then applied once to the test rows. They are on this
    figure because a single accuracy number hides the trade-off a screening
    tool actually has to make: at the arbitrary 0.5 default the model misses
    part of the anemic minority, and buying that sensitivity back costs
    precision along the curve rather than for free.
    """
    y = df["Anemia"].to_numpy()[te]
    prev = float(y.mean())
    # The two Dummy baselines are dropped in one place. This used to be two
    # places - "stratified" filtered out of `order`, then "majority" skipped by a
    # `continue` inside the loop - which also meant the enumerate index that picks
    # the colour was advanced by the skipped model, so the palette had a gap in it
    # wherever majority happened to sort. The horizontal no-skill line below is
    # what a majority-class predictor's precision is, drawn once.
    order = sorted((n for n in fitted if n not in bm.BASELINES),
                   key=lambda n: -bench["models"][n]["pr_auc"])

    fig, ax = plt.subplots(figsize=(COL1 * 1.55, 3.9))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(order):
        p, r, _ = precision_recall_curve(y, fitted[name]["proba"])
        ax.plot(r, p, lw=1.5, color=cmap(i % 10),
                label=f"{MODEL_LABEL[name]}  {bench['models'][name]['pr_auc']:.4f}")
    ax.axhline(prev, ls="--", lw=1.2, color="#333",
               label=f"no-skill baseline  {prev:.4f}")

    # The operating points, from the like-for-like estimator only: putting four
    # markers on all seven curves would be unreadable, and the
    # thresholds in benchmark.json were selected for this one.
    th = bench["models"][LIKE_FOR_LIKE]["thresholds"]
    # The fixed-specificity key is spelled from benchmark.FIXED_SPECIFICITY
    # rather than hardcoded: raising that constant used to make this marker
    # disappear in silence, because th.get("spec90") just returned None.
    spec_key = f"spec{int(bm.FIXED_SPECIFICITY * 100)}"
    assert spec_key in th, f"{spec_key} not in benchmark thresholds: {list(th)}"
    marks = [("default", "o", (6, 10), "left"),
             ("f1", "s", (-6, -16), "right"),
             ("youden", "^", (-8, 8), "right"),
             (spec_key, "D", (-8, -8), "right")]
    for key, mk, off, ha in marks:
        t = th.get(key)
        if t is None:
            continue
        ax.plot(t["test_sensitivity"], t["test_precision"], mk, ms=5.5,
                mfc="white", mec="#111", mew=1.2, zorder=6)
        ax.annotate(f"{key} @{t['threshold']:.2f}\n"
                    f"sens {t['test_sensitivity']:.3f} "
                    f"prec {t['test_precision']:.3f}",
                    (t["test_sensitivity"], t["test_precision"]),
                    textcoords="offset points", xytext=off, fontsize=6.2,
                    color="#111", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#bbb", lw=0.5, alpha=0.9), zorder=7)
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision (PPV)")
    ax.set_title("Precision-recall at L6 (identity-free + demographics)\n"
                 f"markers: {MODEL_LABEL[LIKE_FOR_LIKE]} thresholds chosen on "
                 "validation, applied once to test", fontsize=8.4)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", title="PR-AUC", title_fontsize=7.5,
              fontsize=7, framealpha=0.95)
    save(fig, "fig4_pr_curves")


def fig5_calibration(bench):
    """
    R4-5: are the probabilities usable, or only the labels?

    Read straight out of benchmark.json - the reliability bins were computed
    there, so this figure cannot disagree with the table.
    """
    mods = bench["models"]
    # Hand-ordered so the best-calibrated curves are drawn in the first colours;
    # the two Dummy baselines have no meaningful reliability curve and are named
    # in the right panel instead. Checked against the artefact rather than
    # trusted: an estimator added to benchmark.models() would otherwise be
    # missing from the left panel and from `colour`, and would appear grey in the
    # right panel as though it had deliberately been left out.
    candidates = ("histgb", "xgboost", "rf", "extratrees", "svm_rbf",
                  "logreg", "dtree")
    assert set(candidates) | set(bm.BASELINES) == set(mods), (
        "fig5's model list has drifted from benchmark.json: "
        f"in the artefact but not drawn {sorted(set(mods) - set(candidates) - set(bm.BASELINES))}, "
        f"drawn but not in the artefact {sorted(set(candidates) - set(mods))}")
    show = [n for n in candidates if mods[n]["calibration"]["bin_observed"]]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.4),
                             gridspec_kw={"width_ratios": [1.25, 1],
                                          "wspace": 0.45})
    # One colour per model, shared by both panels: the right panel is already a
    # labelled list of every model, so colouring its bars to match makes it the
    # legend and leaves the reliability curves unobstructed.
    cmap = plt.get_cmap("tab10")
    colour = {n: cmap(i % 10) for i, n in enumerate(show)}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color="#333",
            label="perfect calibration")
    for name in show:
        c = mods[name]["calibration"]
        ax.plot(c["bin_predicted"], c["bin_observed"], marker="o", ms=3.5,
                lw=1.3, color=colour[name], label=MODEL_LABEL[name])
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed anemia fraction")
    ax.set_title("Reliability diagram, L6 (quantile bins)")
    # Only the diagonal needs a key; the models are named in the right panel.
    ax.legend(handles=[ax.lines[0]], loc="lower right", fontsize=6.8,
              framealpha=0.95, handlelength=1.8)

    ax = axes[1]
    names = sorted(mods, key=lambda n: mods[n]["calibration"]["brier"])
    vals = [mods[n]["calibration"]["brier"] for n in names]
    cols = ["#999999" if n not in colour else colour[n] for n in names]
    bars = ax.barh([MODEL_LABEL[n] for n in names], vals, color=cols)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    ax.invert_yaxis()
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_title("Probability quality (colours as left panel)")
    ax.set_xlim(0, max(vals) * 1.22)
    save(fig, "fig5_calibration")


def _cm_panel(ax, y, pred, names, title):
    cm = confusion_matrix(y, pred)
    # A class with no test rows gives a zero row sum. Left as a bare division
    # that is 0/0, which numpy answers with nan and a RuntimeWarning, and the
    # nan then goes to imshow as a blank cell and to the label as "nan%" - a
    # panel that looks broken rather than one that says the class is empty.
    tot = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, tot, out=np.zeros(cm.shape, dtype=float), where=tot > 0)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else "#111")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5, rotation=90, va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(title, fontsize=8.5)
    ax.grid(False)


def fig6_confusion(df, tr, te):
    """
    L1 (submitted) against L6 (defensible), binary and severity, on the same
    test rows. Cells show counts and row-normalised recall, because the majority
    class here is roughly nine rows in ten and the raw counts alone hide the
    minority performance. The exact share is on fig1 and in ablation.json; it is
    not repeated as a literal here, where nothing would check it.
    """
    # Feature lists read off dp.LADDER instead of being re-typed. The previous
    # version spelled "(RBC, MCV, MCH)" and "(RBC, MCV, RDW + demo)" by hand, so
    # a change to the ladder would have relabelled nothing and the panel titles
    # would have named features the model was not given.
    def _feats(rung):
        return ", ".join(FEAT_LABEL.get(c, c) for c in dp.LADDER[rung])

    pretty = {SUBMITTED: f"L1 submitted ({_feats(SUBMITTED)})",
              HEADLINE: f"L6 defensible ({_feats(HEADLINE)})"}
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 6.6))
    specs = [
        ("Anemia", ["Normal", "Anemic"], 0),
        ("Severity", dp.SEVERITY_NAMES_3, 1),
    ]
    for target, names, row in specs:
        y = df[target].to_numpy()
        for col, rung in enumerate([SUBMITTED, HEADLINE]):
            X = df[dp.LADDER[rung]].to_numpy()
            clf = bm.models()["rf"]
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            # Row sums of the confusion matrix, not np.bincount(y[te]). Those are
            # the same numbers only while every class the model predicts also
            # occurs in y_true: confusion_matrix indexes on the union of true and
            # predicted labels, bincount on the range of the true labels alone,
            # so a class predicted but never observed made the two arrays
            # different lengths and the division raise. Using the matrix's own
            # margins is conformable by construction, and matches the percentages
            # _cm_panel prints because it normalises the same way.
            cm = confusion_matrix(y[te], pred)
            tot = cm.sum(axis=1)
            rec = ", ".join(
                f"{v:.2f}" if n else "n/a"
                for v, n in zip(np.divide(np.diag(cm), tot, out=np.zeros(len(tot)),
                                          where=tot > 0), tot))
            _cm_panel(axes[row][col], y[te], pred, names,
                      f"{target} - {pretty[rung]}\nper-class recall: {rec}")
    fig.suptitle("Random Forest confusion matrices, identical test rows "
                 f"(n = {len(te):,})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.6)
    save(fig, "fig6_confusion")


def fig7_importance(df, tr, te, fitted):
    """
    Why the manuscript should not quote mean-decrease-in-impurity.

    MDI counts how often a feature is split on, so it inflates continuous,
    high-cardinality variables and flattens binary ones. Permutation importance
    measures what the model actually loses without the feature.

    Which feature the two disagree about most is computed below and put in the
    title. This docstring used to name sex and pregnancy, which was a guess
    written before the figure was ever generated and is not what the fitted
    model reports - so the prose and the panel could contradict each other with
    nothing to notice.
    """
    rung = HEADLINE
    cols = list(dp.LADDER[rung])
    X = df[cols].to_numpy()
    y = df["Anemia"].to_numpy()
    clf = fitted["rf"]["est"]

    mdi = clf.feature_importances_
    perm = permutation_importance(clf, X[te], y[te], n_repeats=PERM_REPEATS,
                                  random_state=dp.SEED, n_jobs=-1,
                                  scoring="average_precision")

    order = np.argsort(perm.importances_mean)
    lab = [FEAT_LABEL.get(cols[i], cols[i]) for i in order]
    yy = np.arange(len(cols))
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.2), sharey=True)

    axes[0].barh(yy, mdi[order], color="#8172b2")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(lab, fontsize=8)
    axes[0].set_xlabel("mean decrease in impurity")
    axes[0].set_title("MDI (what the manuscript would report)")
    # MDI is a sum of non-negative impurity decreases, so 0 is a real floor here.
    axes[0].set_xlim(0, mdi.max() * 1.22)
    for i, v in enumerate(mdi[order]):
        axes[0].text(v, i, f" {v:.3f}", va="center", fontsize=7)

    pm, ps = perm.importances_mean[order], perm.importances_std[order]
    axes[1].barh(yy, pm, xerr=ps, color="#55a868",
                 error_kw={"lw": 0.9, "capsize": 2})
    axes[1].set_xlabel("drop in PR-AUC when permuted")
    axes[1].set_title(f"Permutation importance ({PERM_REPEATS} repeats)")
    # Unlike MDI, this one is a score DIFFERENCE and is routinely negative for a
    # feature the model does not use: permuting it can help by chance. Starting
    # the axis at 0 clipped those bars and their error bars to nothing, which
    # reads as "no importance" instead of "no signal, and the noise is this wide"
    # - in the panel whose whole point is that MDI overstates weak features.
    lo = float((pm - ps).min()) * 1.22
    axes[1].set_xlim(min(0.0, lo), max(0.0, float((pm + ps).max()) * 1.22))
    for i, (v, s) in enumerate(zip(pm, ps)):
        axes[1].text(v + s, i, f" {v:.3f}", va="center", fontsize=7)

    # State the disagreement the two estimators actually show, rather than the
    # one it would be convenient to claim.
    r_mdi = list(np.argsort(-mdi))
    r_perm = list(np.argsort(-perm.importances_mean))
    moved = sorted(range(len(cols)),
                   key=lambda i: -abs(r_mdi.index(i) - r_perm.index(i)))
    top = moved[0]
    fig.suptitle(
        f"Feature importance at L6: MDI ranks "
        f"{FEAT_LABEL.get(cols[top], cols[top])} #{r_mdi.index(top)+1} of "
        f"{len(cols)}, permutation ranks it #{r_perm.index(top)+1}",
        fontsize=8.5, y=1.02)
    save(fig, "fig7_importance")
    return {c: {"mdi": float(mdi[i]),
                "permutation_mean": float(perm.importances_mean[i]),
                "permutation_sd": float(perm.importances_std[i])}
            for i, c in enumerate(cols)}


def fig8_prevalence(wt):
    """
    The survey figure none of the 20 reviewed papers has: sample prevalence is
    not population prevalence, and the design-based CIs are wide where the
    subgroup is small.
    """
    subs = {k: v for k, v in wt["subgroups"].items() if k != "all"}
    names = list(subs)
    un = np.array([subs[k]["prop_unweighted"] for k in names]) * 100
    w = np.array([subs[k]["prop_weighted"] for k in names]) * 100
    lo = np.array([subs[k]["ci95_logit"][0] for k in names]) * 100
    hi = np.array([subs[k]["ci95_logit"][1] for k in names]) * 100
    n = [subs[k]["n_obs"] for k in names]
    yy = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.9),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.barh(yy + 0.19, un, 0.36, color="#c44e52", label="unweighted sample")
    ax.barh(yy - 0.19, w, 0.36, color="#4c72b0", label="survey weighted")
    ax.errorbar(w, yy - 0.19, xerr=[w - lo, hi - w], fmt="none",
                ecolor="#1a1a1a", elinewidth=0.9, capsize=2.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{k}  (n={v:,})" for k, v in zip(names, n)],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("anemia prevalence (%)")
    ax.set_xlim(0, max(hi.max(), un.max()) * 1.28)
    ax.set_title("Sample vs population prevalence, with design-based 95% CI")
    ax.legend(loc="upper right", framealpha=0.95)

    ov = wt["prevalence"]["Anemia"]
    ax = axes[1]
    ax.grid(False)
    ax.axis("off")
    lines = [
        ("overall unweighted", f"{ov['prop_unweighted']*100:.2f}%"),
        ("overall weighted", f"{ov['prop_weighted']*100:.2f}%"),
        ("design-based SE", f"{ov['se']*100:.2f}"),
        ("95% CI", f"[{ov['ci95_logit'][0]*100:.2f}, "
                   f"{ov['ci95_logit'][1]*100:.2f}]"),
        ("naive binomial SE", f"{ov['se_srs_naive']*100:.2f}"),
        ("design effect", f"{ov['design_effect']:.2f}"),
        ("CIs too narrow by", f"{np.sqrt(ov['design_effect']):.2f}x"),
        ("nominal n", f"{ov['n_obs']:,}"),
        ("effective n", f"{ov['n_effective_design']:,.0f}"),
        ("strata / PSUs", f"{wt['design']['n_strata']} / "
                          f"{wt['design']['total_psus']}"),
        ("degrees of freedom", f"{wt['design']['df']}"),
        ("population represented", f"{wt['design']['population_represented']/1e6:.1f} M"),
        ("anemic population", f"{ov['population_with_outcome']/1e6:.2f} M"),
    ]
    for i, (k, v) in enumerate(lines):
        yv = 0.97 - i * 0.075
        ax.text(0.0, yv, k, fontsize=7.6, va="center")
        ax.text(1.0, yv, v, fontsize=7.6, va="center", ha="right",
                fontweight="bold")
        ax.plot([0, 1], [yv - 0.037, yv - 0.037], lw=0.4, color="#ccc")
    ax.set_title("Design summary")
    save(fig, "fig8_prevalence")

def fig9_contrasts(ab):
    """
    The six pre-declared contrasts, and the three variance scales they could
    have been measured on.

    Left panel: each contrast with its Rao-Wu 95% CI and its Holm-corrected
    p-value. Declaring the family in advance and correcting over it is what
    stops six tests becoming six chances.

    Right panel: the standard error of the same six differences under three
    estimators, as a ratio to the i.i.d. row bootstrap. This is here because
    the intuition "clustering always widens intervals" is wrong in a specific
    and checkable way: the naive n_h-of-n_h cluster bootstrap has variance
    expectation (n_h - 1)/n_h of the truth, which at two PSUs per stratum is
    half, so it comes out NARROWER. A reader can see that rather than be told.
    """
    con = ab["contrasts"]
    vs = ab["variance_scales"]
    n = len(con)
    yy = np.arange(n)
    names = [t["meaning"] for t in con]
    delta = np.array([t["delta_accuracy"] for t in con])
    lo = np.array([t["ci95"][0] for t in con])
    hi = np.array([t["ci95"][1] for t in con])
    sig = [t["significant_holm_05"] for t in con]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 4.0),
                             gridspec_kw={"width_ratios": [1.5, 1],
                                          "wspace": 0.34})
    ax = axes[0]
    ax.axvline(0, lw=1.1, color="#333")
    span = max(hi.max() - lo.min(), 1e-6)
    # Every annotation starts at the same x, clear of the widest interval, so
    # the labels form a column instead of colliding with the zero line.
    xtext = hi.max() + 0.06 * span
    for i in range(n):
        c = "#4c72b0" if sig[i] else "#b0b0b0"
        ax.errorbar(delta[i], yy[i], xerr=[[delta[i] - lo[i]], [hi[i] - delta[i]]],
                    fmt="o", ms=5, color=c, ecolor=c, elinewidth=1.4, capsize=3)
        ax.annotate(f"{delta[i]:+.4f}   Holm p {con[i]['p_holm']:.4f}"
                    f"{'' if sig[i] else '   (null)'}",
                    (xtext, yy[i]), fontsize=6.4, ha="left", va="center",
                    color="#111" if sig[i] else "#666")
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(lo.min() - 0.10 * span, xtext + 0.78 * span)
    ax.set_xlabel(f"accuracy difference, Rao-Wu 95% CI on {vs['design_df']} df",
                  fontsize=7.6)
    n_sig = sum(sig)
    ax.set_title(f"{n_sig} of {n} pre-declared contrasts survive Holm",
                 fontsize=8.5)

    ax = axes[1]
    w = 0.27
    scales = [("iid_row_bootstrap", "i.i.d. rows", "#c44e52"),
              ("naive_cluster_bootstrap", "naive cluster", "#dd8452"),
              ("design", "Rao-Wu (reported)", "#4c72b0")]
    for j, (key, lab, col) in enumerate(scales):
        if key == "design":
            se = np.array([t["se"] for t in con])
        else:
            se = np.array([t[key]["se"] for t in con])
        ratio = se / np.array([t["iid_row_bootstrap"]["se"] for t in con])
        ax.barh(yy + (1 - j) * w, ratio, w, color=col, label=lab)
    ax.axvline(1.0, lw=1.1, color="#333")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"C{i+1}" for i in yy], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("SE relative to the i.i.d. row bootstrap", fontsize=7.6)
    ax.set_title("The variance scale is a choice\n"
                 f"median: naive "
                 f"{vs['median_se_ratio_naive_cluster_over_row']:.2f}x, "
                 f"Rao-Wu {vs['median_se_ratio_raowu_over_row']:.2f}x",
                 fontsize=8.2)
    ax.legend(fontsize=6.4, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False,
              columnspacing=1.0, handlelength=1.2)
    fig.text(0.0, -0.13,
             "C1-C6 are the contrasts in the left panel, top to bottom. The "
             "naive cluster bootstrap is NARROWER than ignoring the design\n"
             "altogether -- its variance expectation is (n_h-1)/n_h of the "
             "truth, one half at two PSUs per stratum -- so it is not a "
             "conservative shortcut.",
             fontsize=6.6, color="#444")
    save(fig, "fig9_contrasts")


def fig10_out_of_cycle(ab):
    """
    The ladder on three NHANES cycles the models were never fitted on.

    Everything here comes from estimators fitted on the 2017-2020 training
    split alone; the earlier cycles are scored, never seen. The point is the
    SIGN, not the size: the identity-present rung beats the identity-free one
    in every cycle, which is what an arithmetic identity predicts, but the size
    of the gap ranges over roughly an order of magnitude across cycles, so the
    right claim is "it reappears", not "it is worth 1.5 points".

    Plotted on the LIFT scale - accuracy minus that cycle's own
    no-information rate - because prevalence differs by cycle and raw accuracy
    would reward whichever cycle happened to have fewer anemic people.
    """
    ext = ab["out_of_cycle"]
    cycles = list(ext)
    show = [SUBMITTED, "L4_identity_free", HEADLINE, "L7_hgb_demo"]
    col = {SUBMITTED: "#c44e52", "L4_identity_free": "#4c72b0",
           HEADLINE: "#55a868", "L7_hgb_demo": "#8172b2"}
    lab = {SUBMITTED: "L1 submitted (identity present)",
           "L4_identity_free": "L4 identity free",
           HEADLINE: "L6 defensible (identity free + demo)",
           "L7_hgb_demo": "L7 hemoglobin handed over"}

    # The primary cohort's held-out test rows, as the right-hand reference bar.
    prim = {r: ab["rungs"][r]["binary"] for r in show}
    nir = ab["majority_baseline_binary"]
    prim_lift = {r: (prim[r]["test_accuracy"] - nir) * 100 for r in show}
    all_cols = cycles + ["2017-2020"]
    xs = np.arange(len(all_cols))

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.9),
                             gridspec_kw={"width_ratios": [1.45, 1],
                                          "wspace": 0.3})
    ax = axes[0]
    w = 0.2
    for j, r in enumerate(show):
        vals = [ext[c]["rungs"][r]["lift_points"] for c in cycles] + \
               [round(prim_lift[r], 2)]
        bars = ax.bar(xs + (j - 1.5) * w, vals, w, color=col[r], label=lab[r],
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.1f", fontsize=5.8, padding=1.5)
    ax.axhline(0, lw=1.1, color="#333")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{c}\nn={ext[c]['n']:,}\nprev {ext[c]['prevalence']*100:.1f}%"
         for c in cycles] +
        [f"2017-2020\nn={ab['n_test']:,}\n"
         f"prev {ab['pr_auc_no_skill']*100:.1f}%\n(held out)"], fontsize=6.4)
    ax.set_ylabel("accuracy points of lift over that\ncycle's own floor",
                  fontsize=7.8)
    ax.set_title("Out-of-cycle replication: fitted on 2017-2020 only",
                 fontsize=8.5)
    ax.legend(fontsize=6.3, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), frameon=False)

    ax = axes[1]
    gap = [ext[c]["rungs"][SUBMITTED]["lift_points"] -
           ext[c]["rungs"]["L4_identity_free"]["lift_points"] for c in cycles]
    gap.append(round(prim_lift[SUBMITTED] - prim_lift["L4_identity_free"], 2))
    bars = ax.bar(xs, gap, 0.55,
                  color=["#4c72b0"] * len(cycles) + ["#333333"],
                  edgecolor="white")
    ax.bar_label(bars, fmt="%+.2f", fontsize=7, padding=2)
    ax.axhline(0, lw=1.1, color="#333")
    ax.axhline(float(np.mean(gap)), ls="--", lw=1.1, color="#c44e52",
               label=f"mean {np.mean(gap):+.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("-", "\n-") for c in all_cols], fontsize=6.4)
    ax.set_ylabel("L1 minus L4, lift points", fontsize=7.8)
    # min(0, ...) on the floor. The title two lines below counts how many cycles
    # the gap is POSITIVE in, i.e. it is written to survive a negative gap - but
    # a y-axis starting at 0 would clip that bar to nothing, so the panel would
    # show a missing cycle while the title reported a negative one. The 2015-2016
    # margin is +0.23 lift points, close enough to zero that this is one cycle
    # away, not a hypothetical.
    ax.set_ylim(min(0.0, min(gap) * 1.32), max(0.0, max(gap) * 1.32))
    # Stated as sign plus range rather than "it replicates", because one cycle
    # puts the gap at +0.23 and rounding that into an average would be the
    # same kind of favourable summary this paper is objecting to.
    n_pos = sum(1 for g in gap if g > 0)
    ax.set_title(f"The identity gap: positive in {n_pos} of {len(gap)} cycles\n"
                 f"but the size varies, {min(gap):+.2f} to {max(gap):+.2f}",
                 fontsize=8.2)
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.95)
    save(fig, "fig10_out_of_cycle")


def main():
    ab = load("ablation")
    bench = load("benchmark")
    wt = load("weighted")
    b6 = bench["binary"][HEADLINE]

    df, att = dp.load_cohort()
    # Three-way split, the same one ablation.py and benchmark.py use: the
    # figures must not be drawn from rows either script tuned on.
    tr, va, te = dp.split3(df)
    print(f"cohort {att['n_analysis']:,}   train {len(tr):,}   "
          f"val {len(va):,}   test {len(te):,}")
    print(f"writing to {FIGURES}\n")

    fig1_flowchart(ab, wt, bench)
    fig2_ladder(ab)
    fig3_decomposition(ab)

    # One refit at L6, shared by the PR curves and the importance panel, so the
    # two figures are guaranteed to describe the same fitted models.
    _, _, fitted = refit(df, tr, te, HEADLINE)
    fig4_pr_curves(df, tr, te, fitted, b6)
    fig5_calibration(b6)
    fig6_confusion(df, tr, te)
    imp = fig7_importance(df, tr, te, fitted)
    fig8_prevalence(wt)
    fig9_contrasts(ab)
    fig10_out_of_cycle(ab)

    with open(RESULTS / "importance.json", "w", encoding="utf-8") as fh:
        # allow_nan=False: json.dump would otherwise write bare NaN/Infinity
        # tokens, which are not valid JSON. A permutation SD of nan would mean
        # something went wrong upstream and should stop the run, not be written.
        json.dump({"rung": HEADLINE,
                   "scoring": "average_precision",
                   "n_repeats": PERM_REPEATS,
                   "features": imp}, fh, indent=1, allow_nan=False)

    print("\nMDI vs permutation at L6")
    print(f"  {'feature':<12}{'MDI':>9}{'rank':>6}{'perm':>9}{'rank':>6}"
          f"{'perm SD':>9}")
    r_mdi = sorted(imp, key=lambda c: -imp[c]["mdi"])
    r_perm = sorted(imp, key=lambda c: -imp[c]["permutation_mean"])
    for c in r_perm:
        v = imp[c]
        print(f"  {c:<12}{v['mdi']:9.4f}{r_mdi.index(c)+1:6d}"
              f"{v['permutation_mean']:9.4f}{r_perm.index(c)+1:6d}"
              f"{v['permutation_sd']:9.4f}")
    flips = [(c, r_mdi.index(c) + 1, r_perm.index(c) + 1) for c in imp
             if r_mdi.index(c) != r_perm.index(c)]
    if flips:
        print("\n  The two estimators do not agree on the ordering:")
        for c, a, b in sorted(flips, key=lambda t: -abs(t[1] - t[2])):
            print(f"    {c:<12} MDI #{a}  ->  permutation #{b}")
        print("  MDI counts splits, so it rewards continuous features and")
        print("  flattens binary ones. Permutation measures what the model")
        print("  actually loses. A paper that quotes MDI is quoting the first.")

    l4 = ab["rungs"]["L4_identity_free"]["binary"]["test_accuracy"]
    l6 = ab["rungs"][HEADLINE]["binary"]["test_accuracy"]
    print(f"\n  Adding age, sex and pregnancy to L4 is worth "
          f"{(l6 - l4)*100:.2f} accuracy points ({l4:.4f} -> {l6:.4f}),")
    print(f"  which MDI scores at "
          f"{sum(imp[c]['mdi'] for c in (dp.AGE, dp.SEX, dp.PREG)):.3f} of 1.000 "
          "in total.")

    print(f"\nwrote {RESULTS / 'importance.json'}")
    print(f"10 figures in {FIGURES} (PNG at 300 dpi + PDF)")


if __name__ == "__main__":
    main()
