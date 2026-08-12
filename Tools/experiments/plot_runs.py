"""Plot training loss and validation ADE/FDE per epoch for two or more runs.

Reads the per-epoch lines train_il prints, so it works on any run's log without
a separate metrics export:

    Epoch 3/20 loss=0.1954 ... val_ADE=1.1794 val_FDE=3.4812 ...

Three small multiples, never a dual-axis chart: loss (~0.1-0.3), ADE and FDE
(metres) live on different scales, and a second y-axis makes their relationship
an artefact of the scaling rather than of the data.

Colours are the first slots of a categorical palette validated for colour-vision
deficiency (worst pair deltaE 24.7 protan / 33.6 normal vision). Add hues from
SERIES_COLORS in order; past four series, facet instead.

Usage:
    python Tools/experiments/plot_runs.py run_a.log run_b.log -o comparison.png
    python Tools/experiments/plot_runs.py *.log --labels residual deformable
"""

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+loss=([0-9.]+).*?val_ADE=([0-9.]+)\s+val_FDE=([0-9.]+)"
)

# Categorical slots 1-4, in fixed order. Never cycle or generate a new hue.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#78776f"
GRID = "#e2e2dc"

PANELS = [
    ("loss", "Training loss", "SmoothL1 (accel/curvature)", "{:.4f}"),
    ("ade", "Validation ADE @ 3s", "metres", "{:.2f}"),
    ("fde", "Validation FDE @ 3s", "metres", "{:.2f}"),
]


def parse_log(path):
    """Return {'epochs': [...], 'loss': [...], 'ade': [...], 'fde': [...]}."""
    series = {"epochs": [], "loss": [], "ade": [], "fde": []}
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        series["epochs"].append(int(m.group(1)))
        series["loss"].append(float(m.group(2)))
        series["ade"].append(float(m.group(3)))
        series["fde"].append(float(m.group(4)))
    if not series["epochs"]:
        raise SystemExit(f"no epoch lines found in {path}")
    # Best epoch = lowest validation ADE, matching train_il's checkpoint selector.
    series["best_idx"] = min(
        range(len(series["ade"])), key=lambda i: series["ade"][i]
    )
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="train_il log files")
    ap.add_argument("--labels", nargs="*", help="series names (default: file stems)")
    ap.add_argument("-o", "--output", default="run_comparison.png")
    ap.add_argument("--title", default="AutoE2E run comparison")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--provenance", default="", help="digest line printed at the bottom")
    args = ap.parse_args()

    if len(args.logs) > len(SERIES_COLORS):
        raise SystemExit(
            f"{len(args.logs)} runs exceeds the {len(SERIES_COLORS)}-hue cap; "
            "facet into separate figures instead of generating hues"
        )

    labels = args.labels or [pathlib.Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must have one entry per log file")
    runs = {
        lab: {**parse_log(p), "color": SERIES_COLORS[i]}
        for i, (lab, p) in enumerate(zip(labels, args.logs))
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.05, right=0.99, wspace=0.24)

    for ax, (key, title, ylab, vfmt) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c9c2")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        # Headroom so a below-marker label never touches the tick row.
        ax.margins(y=0.16)

        for lab, r in runs.items():
            ax.plot(
                r["epochs"], r[key], color=r["color"], linewidth=2,
                solid_capstyle="round", label=lab, zorder=3,
            )
            bi = r["best_idx"]
            ax.plot(
                r["epochs"][bi], r[key][bi], "o", color=r["color"], markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
            )

        # Runs often share a best epoch, so stagger the labels by value order.
        ranked = sorted(runs.values(), key=lambda r: r[key][r["best_idx"]], reverse=True)
        offsets = [14, -20, 30, -36][: len(ranked)]
        for r, dy in zip(ranked, offsets):
            bi = r["best_idx"]
            ax.annotate(
                vfmt.format(r[key][bi]), (r["epochs"][bi], r[key][bi]),
                textcoords="offset points", xytext=(6, dy), ha="left",
                fontsize=9, fontweight="bold", color=r["color"], zorder=5,
            )

        last = max(r["epochs"][-1] for r in runs.values())
        if any(r["epochs"][-1] == last for r in runs.values()):
            ax.axvline(last, color=MUTED, linewidth=1, linestyle=(0, (3, 3)),
                       alpha=0.65, zorder=1)
            ax.text(last - 0.1, ax.get_ylim()[1], "last epoch ", ha="right",
                    va="top", fontsize=8, color=MUTED)

        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8, loc="left")
        ax.set_xlabel("epoch", fontsize=9.5, color=INK_2)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK_2)
        ax.set_xticks(sorted({e for r in runs.values() for e in r["epochs"]}))
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.text(0.05, 0.955, args.title, fontsize=15, fontweight="bold", color=INK)
    if args.subtitle:
        fig.text(0.05, 0.905, args.subtitle, fontsize=9.5, color=INK_2)
    if args.provenance:
        fig.text(0.05, 0.045, args.provenance, fontsize=7.5, color=MUTED,
                 family="monospace")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", bbox_to_anchor=(0.99, 0.985),
               frameon=False, fontsize=10, labelcolor=INK_2,
               ncol=len(lbls), handlelength=1.6)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    for lab, r in runs.items():
        bi = r["best_idx"]
        print(
            f"  {lab:<16} best epoch {r['epochs'][bi]:>2}  "
            f"ADE {r['ade'][bi]:.4f}  FDE {r['fde'][bi]:.4f}"
        )


if __name__ == "__main__":
    main()
