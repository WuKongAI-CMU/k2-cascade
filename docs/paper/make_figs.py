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


if __name__ == "__main__":
    fig_arms()
    fig_layers()
    fig_distance()
    fig_qa()
    fig_retention()
