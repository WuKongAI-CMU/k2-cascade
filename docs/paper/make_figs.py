"""Paper figures for docs/paper/main.tex.

Run from the repo root:
    uv run --with matplotlib python docs/paper/make_figs.py

Reads analysis/cloud/*.json and writes docs/paper/figs/fig_*.pdf (vector) plus
PNG previews. Every number is read from the JSON files; nothing is typed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "cloud"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette.
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
    "black": "#000000",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

SINGLE = 3.25
DOUBLE = 6.75
CHANCE = 0.125


def load(name: str) -> dict:
    with open(DATA / f"{name}.json") as f:
        return json.load(f)


def acc(name: str, arm: str) -> float:
    return load(name)[arm]["acc"]


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def edge_label(ax, y, text, color):
    """Label a horizontal reference line just outside the right axis edge."""
    ax.text(1.01, y, text, transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=6.5, color=color, clip_on=False)


def bar_labels(ax, bars, vals, fmt="{:.0f}", dy=0.012, fontsize=7):
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            fmt.format(100 * v),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# --------------------------------------------------------------------------- (a)
def fig_arms() -> None:
    seeds = [f"prog_noma_mlp_top3_s1500_seed{i}" for i in range(3)]
    base = load(seeds[0])  # none/text/raw arms are projector-independent
    trained = np.array([acc(s, "project") for s in seeds])
    deranged = np.array([acc(s, "derange") for s in seeds])
    labels = [
        "none",
        "text",
        "raw",
        "ridge top-3",
        "trained",
        "deranged cache",
        "deranged-sender ctrl",
    ]
    vals = [
        base["none"]["acc"],
        base["text"]["acc"],
        base["raw"]["acc"],
        acc("prog_noma_ridge_top3", "project"),
        trained.mean(),
        deranged.mean(),
        acc("noma_main_ctrl_derange_s1500", "project"),
    ]
    errs = [0, 0, 0, 0, trained.std(ddof=1), deranged.std(ddof=1), 0]
    colours = [
        C["grey"],
        C["black"],
        C["grey"],
        C["sky"],
        C["blue"],
        C["vermillion"],
        C["orange"],
    ]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    x = np.arange(len(vals))
    bars = ax.bar(
        x, vals, yerr=errs, color=colours, width=0.7, capsize=2,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
    )
    ax.axhline(CHANCE, ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(ax, CHANCE, "chance", C["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Noma accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for b, v, e in zip(bars, vals, errs):
        txt = f"{100 * v:.0f}" if e == 0 else f"{100 * v:.0f}±{100 * e:.0f}"
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.015, txt,
                ha="center", va="bottom", fontsize=6.5)
    save(fig, "fig_arms")


# --------------------------------------------------------------------------- (b)
def fig_layers() -> None:
    bands = ["0-5", "0-11", "0-17", "12-23", "24-35", "30-35", "18-35"]
    vals = [acc(f"noma_layers{b}_mlp", "project") for b in bands]
    derange = [acc(f"noma_layers{b}_mlp", "derange") for b in bands]
    assert max(derange) - min(derange) < 1e-9, derange
    bands.append("all")
    vals.append(acc("noma_main_mlp_top3_s1500_seed0", "project"))
    labels = [b.replace("-", "–") for b in bands]

    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    x = np.arange(len(vals))
    colours = [C["sky"]] * 3 + [C["blue"]] * 4 + [C["black"]]
    bars = ax.bar(x, vals, color=colours, width=0.7)
    ax.axhline(derange[0], ls="-", lw=0.8, color=C["vermillion"], zorder=0,
               label="deranged cache, all layers")
    ax.axhline(CHANCE, ls="--", lw=0.8, color=C["grey"], zorder=0, label="chance")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("receiver layers given this episode's cache")
    ax.set_ylabel("Noma accuracy")
    ax.set_ylim(0, 0.75)
    bar_labels(ax, bars, vals)
    save(fig, "fig_layers")


# --------------------------------------------------------------------------- (c)
def fig_distance() -> None:
    pads = [0, 64, 128, 256, 512]

    def stem(pad: int, proj: str) -> str:
        return f"noma_{'main' if pad == 0 else f'pad{pad}'}_{proj}"

    series = [
        ("trained", "mlp_top3_s1500_seed0", C["blue"], "o"),
        ("ridge top-3", "ridge_top3", C["sky"], "s"),
        ("deranged-sender ctrl", "ctrl_derange_s1500", C["orange"], "^"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    for label, proj, col, mk in series:
        ys = [acc(stem(p, proj), "project") for p in pads]
        ax.plot(range(len(pads)), ys, marker=mk, ms=4, lw=1.2, color=col, label=label)
    ax.axhline(CHANCE, ls="--", lw=0.8, color=C["grey"], zorder=0, label="chance")
    ax.set_xticks(range(len(pads)))
    ax.set_xticklabels([str(p) for p in pads])
    ax.set_xlabel("filler tokens between facts and question")
    ax.set_ylabel("Noma accuracy")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="center right", frameon=False, handlelength=1.6)
    save(fig, "fig_distance")


# --------------------------------------------------------------------------- (d)
def fig_qa() -> None:
    none = load("qa_none")
    mlp = load("qa_mlp_seed2")
    labels = [
        "none",
        "text",
        "raw",
        "ridge top-3",
        "trained",
        "deranged-sender ctrl",
        "trained, deranged passage",
    ]
    vals = [
        none["none"]["f1"],
        none["text"]["f1"],
        none["raw"]["f1"],
        load("qa_ridge_top3")["project"]["f1"],
        mlp["project"]["f1"],
        load("qa_ctrl_derange")["project"]["f1"],
        mlp["derange"]["f1"],
    ]
    colours = [
        C["grey"], C["black"], C["grey"], C["sky"], C["blue"], C["orange"], C["vermillion"],
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colours, width=0.7)
    ax.axhline(vals[0], ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(ax, vals[0], "question\nonly", C["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("SQuAD token-F1")
    ax.set_ylim(0, 0.68)
    bar_labels(ax, bars, vals, fmt="{:.1f}")
    save(fig, "fig_qa")


# --------------------------------------------------------------------------- (e)
def fig_retention() -> None:
    projs = [
        ("ridge top-3", "ridge_top3", C["sky"]),
        ("trained", "mlp_top3_s1500_seed0", C["blue"]),
        ("deranged-\nsender ctrl", "ctrl_derange_s1500", C["orange"]),
    ]
    ret = [load(f"ret_fresh_{p}")["retention"] for _, p, _ in projs]
    ret_d = [load(f"ret_fresh_{p}")["retention_derange"] for _, p, _ in projs]
    noma = [acc(f"noma_main_{p}", "project") for _, p, _ in projs]
    noma_d = [acc(f"noma_main_{p}", "derange") for _, p, _ in projs]
    names = [n for n, _, _ in projs]
    cols = [c for _, _, c in projs]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(projs))
    w = 0.36

    # left: continuation retention
    b1 = axl.bar(x - w / 2, ret, w, color=cols, label="sender reads the text")
    b2 = axl.bar(x + w / 2, ret_d, w, color=cols, alpha=0.45, hatch="////",
                 edgecolor="white", linewidth=0, label="sender reads other text (derange)")
    axl.axhline(0, lw=0.6, color=C["black"])
    axl.axhline(1.0, ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(axl, 1.0, "oracle", C["grey"])
    axl.set_xticks(x)
    axl.set_xticklabels(names)
    axl.set_ylabel("continuation retention")
    axl.set_ylim(-0.8, 1.45)
    for bars, vs in ((b1, ret), (b2, ret_d)):
        for b, v in zip(bars, vs):
            y = v + 0.03 if v >= 0 else v - 0.03
            axl.text(b.get_x() + b.get_width() / 2, y, f"{v:.2f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=6.5)
    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C["grey"]),
        plt.Rectangle((0, 0), 1, 1, color=C["grey"], alpha=0.45, hatch="////",
                      edgecolor="white", linewidth=0),
    ]
    axl.legend(leg_handles, ["cache from the text", "cache from other text"],
               loc="lower right", frameon=False, handlelength=1.4)

    # right: Noma accuracy
    b3 = axr.bar(x - w / 2, noma, w, color=cols)
    b4 = axr.bar(x + w / 2, noma_d, w, color=cols, alpha=0.45, hatch="////",
                 edgecolor="white", linewidth=0)
    axr.axhline(CHANCE, ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(axr, CHANCE, "chance", C["grey"])
    axr.set_xticks(x)
    axr.set_xticklabels(names)
    axr.set_ylabel("Noma accuracy")
    axr.set_ylim(0, 0.8)
    for bars, vs in ((b3, noma), (b4, noma_d)):
        bar_labels(axr, bars, vs)
    save(fig, "fig_retention")


# --------------------------------------------------------------------------- (f)
def fig_confidence() -> None:
    """Left: AUROC of the receiver's first-token entropy against the sender's
    semantic entropy, per arm and passage variant.  Right: P(gold) across the
    three variants.  Arms text/verbal/cache/deranged come from the verbal run;
    zero and moment-matched random come from the earlier run on the same 377
    items (the shared arms are asserted equal)."""
    with open(DATA / "verbal" / "confidence_verbal_mlp_seed2.json") as f:
        V = json.load(f)
    with open(DATA / "conf" / "confidence.json") as f:
        Z = json.load(f)
    assert V["n_kept"] == Z["n_kept"] == 377
    variants = ["clean", "contradicted", "removed"]
    for v in variants:
        for arm in ("text", "project", "derange"):
            assert abs(V["sender_tracking"][v][f"auroc_{arm}"]
                       - Z["sender_tracking"][v][f"auroc_{arm}"]) < 1e-9
    arms = [
        ("text", "text", V, C["black"], "o"),
        ("verbal", "text + confidence word", V, C["green"], "D"),
        ("project", "mapped cache", V, C["blue"], "s"),
        ("derange", "deranged cache", V, C["vermillion"], "^"),
        ("zero", "zero cache", Z, C["grey"], "v"),
        ("random", "random cache", Z, C["purple"], "x"),
    ]

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"wspace": 0.28, "width_ratios": [1.45, 1]}
    )

    # left: grouped AUROC bars
    n = len(arms)
    w = 0.8 / n
    x = np.arange(len(variants))
    for i, (key, label, src, col, _) in enumerate(arms):
        ys = [src["sender_tracking"][v][f"auroc_{key}"] for v in variants]
        bars = axl.bar(x + (i - (n - 1) / 2) * w, ys, w, color=col, label=label)
        for b, y in zip(bars, ys):
            axl.text(b.get_x() + b.get_width() / 2, y + 0.004, f"{y:.2f}".lstrip("0"),
                     ha="center", va="bottom", fontsize=5.2, rotation=90)
    axl.axhline(0.5, ls="--", lw=0.8, color=C["grey"], zorder=0)
    axl.set_xticks(x)
    axl.set_xticklabels(["clean", "contradicted", "answer removed"])
    axl.set_ylabel("AUROC vs sender semantic entropy")
    axl.set_ylim(0.4, 0.8)
    axl.set_yticks([0.4, 0.5, 0.6, 0.7])
    axl.set_yticklabels(["0.4", "0.5\nchance", "0.6", "0.7"])
    axl.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.2,
               columnspacing=0.8, fontsize=6.3)

    # right: P(gold) across variants
    for key, label, src, col, mk in arms[:4]:
        ys = [src["per_variant"][v][key]["p_gold"] for v in variants]
        axr.plot(range(3), ys, marker=mk, ms=4, lw=1.2, color=col, label=label)
    q = V["per_variant"]["clean"]["none"]["p_gold"]
    axr.axhline(q, ls=":", lw=0.8, color=C["grey"], zorder=0)
    edge_label(axr, q, "question\nonly", C["grey"])
    axr.set_xticks(range(3))
    axr.set_xticklabels(["clean", "contradicted", "removed"])
    axr.set_ylabel("receiver P(gold answer)")
    axr.set_ylim(0, 0.5)
    axr.legend(loc="upper right", frameon=False, handlelength=1.6, fontsize=6.3)
    save(fig, "fig_confidence")


# --------------------------------------------------------------------------- (g)
def fig_pairs() -> None:
    pairs = [
        ("3.7B\n$\\rightarrow$ 7B", "prog_noma_ridge_top3", "prog_noma_mlp_top3_s1500_seed0"),
        ("7B\n$\\rightarrow$ 3.7B", "fu_noma_rev_ridge_top3", "fu_noma_rev_mlp_top3"),
        ("Qwen3-4B\n$\\rightarrow$ 7B", "xfamily/xf_noma_ridge_qwen3-4b", "xfamily/xf_noma_mlp_qwen3-4b"),
        ("0.9B\n$\\rightarrow$ 7B", "xfamily/xf_noma_ridge_k2-horizon-0_9b", "xfamily/xf_noma_mlp_k2-horizon-0_9b"),
    ]
    ridge = [acc(r, "project") for _, r, _ in pairs]
    trained = [acc(t, "project") for _, _, t in pairs]
    for _, r, t in pairs:
        assert load(r)["n"] == load(t)["n"] == 300

    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    x = np.arange(len(pairs))
    w = 0.38
    b1 = ax.bar(x - w / 2, ridge, w, color=C["sky"], label="ridge top-3 (closed form)")
    b2 = ax.bar(x + w / 2, trained, w, color=C["blue"], label="+ trained residual")
    ax.axhline(CHANCE, ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(ax, CHANCE, "chance", C["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels([p for p, _, _ in pairs], fontsize=7)
    ax.set_xlabel("sender $\\rightarrow$ receiver (K2-Horizon unless named)")
    ax.set_ylabel("Noma accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_labels(ax, b1, ridge)
    bar_labels(ax, b2, trained)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)
    save(fig, "fig_pairs")


# --------------------------------------------------------------------------- (h)
# label text, x offset (pt), y offset (pt), horizontal alignment
COST_LABELS = {
    "none": ("full, bf16", -4, -7, "right"),
    "int8": ("int8", 0, -8, "center"),
    "int4": ("int4", 0, -8, "center"),
    "rank=32": ("rank 32", 5, -2, "left"),
    "rank=16": ("rank 16", 5, -2, "left"),
    "rank=8": ("rank 8", 0, -8, "center"),
    "layers=12-35,fill=mean": ("layers 12–35, mean fill", 0, 6, "center"),
    "layers=18-35,fill=mean": ("layers 18–35, mean fill", 5, 1, "left"),
    "layers=18-35,fill=mean,int4": ("18–35, mean fill, int4", 4, 4, "left"),
    "layers=24-35,fill=mean": ("layers 24–35,\nmean fill", 0, -12, "center"),
    "heads=4,fill=mean": ("4 heads, mean fill", 5, -2, "left"),
    "heads=4": ("4 heads; layers 18–35 (zeros)", 5, 4, "left"),
    "layers=18-35": None,  # same point as heads=4; one label covers both
    "heads=2": ("2 heads; 18–35 + int8 (zeros)", 5, -5, "left"),
    "layers=18-35,int8": None,  # same point as heads=2
}


def _cost_family(spec: str) -> str:
    if "fill=mean" in spec:
        return "fill"
    if spec.startswith("rank"):
        return "rank"
    if spec.startswith("heads") or spec.startswith("layers"):
        return "zeros"
    return "quant"


def fig_cost() -> None:
    files = sorted((DATA / "compress").glob("cmp_qa_*.json")) + sorted(
        (DATA / "compress2").glob("cmp_qa_*.json"))
    pts = []
    for fp in files:
        with open(fp) as f:
            j = json.load(f)
        spec = j["compress"]["spec"]
        pts.append((spec, j["compress"]["bytes_per_token"] / 1024, j["project"]["f1"]))
    assert {s for s, _, _ in pts} == set(COST_LABELS), {s for s, _, _ in pts} ^ set(COST_LABELS)
    text_f1 = load("qa_none")["text"]["f1"]
    none_f1 = load("qa_none")["none"]["f1"]

    fam = {
        "quant": ("full / quantised", C["blue"], "o"),
        "rank": ("low-rank truncation", C["green"], "s"),
        "zeros": ("dropped, zeros", C["vermillion"], "^"),
        "fill": ("dropped, mean fill", C["orange"], "D"),
    }
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for key, (label, col, mk) in fam.items():
        sub = [(x, y) for s, x, y in pts if _cost_family(s) == key]
        ax.scatter([x for x, _ in sub], [y for _, y in sub], s=16, color=col,
                   marker=mk, label=label, zorder=3, linewidths=0)
    for spec, x, y in pts:
        lab = COST_LABELS[spec]
        if lab is None:
            continue
        text, dx, dy, ha = lab
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=5.8)
    ax.axhline(text_f1, ls="--", lw=0.8, color=C["black"], zorder=0)
    edge_label(ax, text_f1, "receiver\nreads text", C["black"])
    ax.axhline(none_f1, ls="--", lw=0.8, color=C["grey"], zorder=0)
    edge_label(ax, none_f1, "question\nonly", C["grey"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128])
    ax.set_xticklabels(["16", "32", "64", "128"])
    ax.set_xlim(13, 190)
    ax.set_xlabel("message size, KB per prefix token (log scale)")
    ax.set_ylabel("SQuAD token-F1")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, fontsize=6,
              bbox_to_anchor=(-0.01, 0.84))
    save(fig, "fig_cost")


# --------------------------------------------------------------------------- (i)
def fig_policy() -> None:
    """F1 against latency (ms per item) as the handoff rate goes from 0 to 1,
    for the three handoff arms under the probe trigger (solid) and the oracle
    trigger (dashed).  Data: analysis/cloud/cascade/curves.json."""
    with open(DATA / "cascade" / "curves.json") as f:
        J = json.load(f)
    arms = [
        ("text", "text handoff", C["black"], "o"),
        ("project", "mapped cache", C["blue"], "s"),
        ("verbal", "answer + confidence word", C["green"], "D"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    for key, label, col, mk in arms:
        for trig, ls, lw in (("probe", "-", 1.2), ("oracle", "--", 0.9)):
            pts = J["curves"][f"{key}/{trig}"]
            xs = [1000 * p["cost"] for p in pts]
            ys = [p["f1"] for p in pts]
            ax.plot(xs, ys, ls=ls, lw=lw, color=col, marker=mk if trig == "probe" else None,
                    ms=3, label=label if trig == "probe" else None)
    sender = J["sender_f1"]
    ax.axhline(sender, ls=":", lw=0.8, color=C["grey"], zorder=0)
    edge_label(ax, sender, "sender\nalone", C["grey"])
    ax.set_xlabel("latency, ms per item (A100)")
    ax.set_ylabel("SQuAD token-F1")
    ax.set_ylim(0.44, 0.60)
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=C["grey"], ls="-", lw=1.2),
          plt.Line2D([], [], color=C["grey"], ls="--", lw=0.9)]
    l += ["probe trigger", "oracle trigger"]
    ax.legend(h, l, loc="lower right", frameon=False, handlelength=1.6, fontsize=6.3, ncol=2,
              columnspacing=0.8)
    save(fig, "fig_policy")


if __name__ == "__main__":
    fig_arms()
    fig_layers()
    fig_distance()
    fig_qa()
    fig_retention()
    fig_confidence()
    fig_pairs()
    fig_cost()
    fig_policy()
