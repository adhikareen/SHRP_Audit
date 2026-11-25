"""CSL-Daily Tripwire figure — three panels, spans both columns (figure*).

Design priorities (v2, post-review):
  - No element overlaps another. Legends live OUTSIDE the data area.
  - Panel-to-panel breathing room. Each panel reads at a glance.
  - Annotations are placed in empty regions, not on top of marks.

Panel A: Horizontal bar chart of CSL-Daily test WER. Three groups:
         published backbones (gray), dev-conditioned probes (red, hatched),
         cross-arch MBR (green).  A vertical dashed line marks the
         best published WER (SCOPE).
Panel B: Nearest-neighbour gloss edit-distance histograms, three datasets
         overlaid on a symlog y-axis (so the CSL spike at distance 0 and the
         PHX/PHX-T smooth humps are both visible).
Panel C: Heatmap of Delta-WER (method x dataset).  Red = worse than greedy,
         blue = better than greedy.  Only cross-arch MBR is blue on PHX/PHX-T.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--nn-dist-dir", type=Path, default=ROOT / "data" / "diagnostics",
                help="folder holding nn_dist_{csl,phx,phxT}.npy")
NN_DIR = ap.parse_args().nn_dist_dir
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.7,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.7,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})

fig = plt.figure(figsize=(7.5, 5.2))
# Outer GridSpec carves out a top legend strip and a bottom panels strip.
outer = fig.add_gridspec(
    2, 1,
    height_ratios=[0.08, 0.92],
    left=0.005, right=0.995, top=0.98, bottom=0.125,
    hspace=0.30,
)
# Panels strip — generous wspace so panel titles never touch.
panels_gs = outer[1].subgridspec(
    1, 3,
    width_ratios=[1.85, 1.45, 1.05],
    wspace=0.62,
)
axA = fig.add_subplot(panels_gs[0])
axB = fig.add_subplot(panels_gs[1])
axC = fig.add_subplot(panels_gs[2])
# Legend strip is a dummy axis we draw into for the global Panel-A legend
ax_leg = fig.add_subplot(outer[0])
ax_leg.axis("off")


# =====================================================================
# Panel A — horizontal bar tripwire on CSL-Daily
# =====================================================================
# (label, WER on CSL-Daily, kind)
SYSTEMS = [
    ("Dev-LM\nrescore",                 14.20, "devcond"),
    ("BoW frozen-emb\n(20-seed ens.)",  16.92, "devcond"),
    ("SCOPE\n(arXiv '24)",              23.10, "pub"),
    ("Online CSLR\n(EMNLP '24)",        24.40, "pub"),
    ("TwoStream-SLR\n(NeurIPS '22)",    25.30, "pub"),
    ("SlowFastSign\n(ICASSP '24)",      25.50, "pub"),
    ("Cross-arch MBR\n(control)",       25.90, "nodev"),
    ("SignGraph\n(CVPR '24)",           26.40, "pub"),
    ("CorrNet+\n(arXiv '24)",           28.60, "pub"),
    ("CorrNet\n(CVPR '23)",             30.10, "pub"),
]
# Sort ascending so best WER is at the top of the chart
SYSTEMS.sort(key=lambda x: x[1])

KIND_COLOR  = {"pub": "#9aa0a6", "devcond": "#c0392b", "nodev": "#1e8449"}
KIND_HATCH  = {"pub": None,      "devcond": "//",      "nodev": None}
KIND_TEXT   = {"pub": "#444444", "devcond": "#7b241c", "nodev": "#145a32"}

y_pos = np.arange(len(SYSTEMS))
for i, (lab, wer, kind) in enumerate(SYSTEMS):
    axA.barh(
        i, wer, height=0.66,
        color=KIND_COLOR[kind], edgecolor="#222222",
        linewidth=0.6, hatch=KIND_HATCH[kind], zorder=3,
    )
    axA.text(
        wer + 0.35, i, f"{wer:.1f}", va="center", ha="left",
        fontsize=8.5, color=KIND_TEXT[kind], fontweight="bold",
    )

axA.set_yticks(y_pos)
axA.set_yticklabels([s[0] for s in SYSTEMS], fontsize=7.8, linespacing=0.95)
axA.set_xlabel("CSL-Daily test WER (%)   $\\leftarrow$ lower is better",
               fontsize=8.7, labelpad=2)
axA.set_xlim(0, 33)
axA.set_ylim(-0.6, len(SYSTEMS) + 0.1)
axA.set_title("(a) CSL-Daily test WER",
              loc="left", fontweight="bold", pad=5, fontsize=9.5)
axA.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
axA.tick_params(axis="y", length=0)

# Vertical reference line at the published best WER (SCOPE = 23.1). Label is
# drawn at the TOP of the vertical line in the empty region above the topmost
# bar, so it never collides with any bar value text.
sota_wer = 23.10
axA.axvline(sota_wer, color="#c0392b", lw=0.9, ls="--", alpha=0.45, zorder=1)
axA.text(
    sota_wer, len(SYSTEMS) - 0.30,
    "  SCOPE = 23.1  (best published)",
    fontsize=7.0, color="#7b241c", style="italic",
    ha="left", va="bottom",
)


# =====================================================================
# Panel B — NN edit-distance histograms (symlog y)
# =====================================================================
nn_csl  = np.load(NN_DIR / "nn_dist_csl.npy")
nn_phx  = np.load(NN_DIR / "nn_dist_phx.npy")
nn_phxT = np.load(NN_DIR / "nn_dist_phxT.npy")

BIN_EDGES = np.linspace(0, 1.0, 21)  # 20 bins, width 0.05

def plot_hist(ax, data, color, label, ls="-", lw=1.6, fill=False):
    counts, _ = np.histogram(data, bins=BIN_EDGES)
    frac = counts / counts.sum()
    ax.step(np.r_[BIN_EDGES[0], BIN_EDGES[:-1]], np.r_[frac[0], frac],
            where="post", color=color, lw=lw, ls=ls, label=label, zorder=3)
    if fill:
        ax.fill_between(BIN_EDGES[:-1], frac, step="post", alpha=0.20,
                        color=color, zorder=2)

plot_hist(axB, nn_csl,  "#c0392b", "CSL-Daily",  fill=True)
plot_hist(axB, nn_phx,  "#1f4e79", "PHX-2014",   ls="--", lw=1.8)
plot_hist(axB, nn_phxT, "#117a3d", "PHX-2014T",  ls="-.", lw=2.0)

axB.set_yscale("symlog", linthresh=0.05, linscale=0.5)
axB.set_xlabel("NN gloss edit distance (test $\\rightarrow$ dev)",
               fontsize=8.7, labelpad=2)
axB.set_ylabel("Fraction of test samples", fontsize=8.7, labelpad=2)
axB.set_title("(b) Dev/test sentence proximity",
              loc="left", fontweight="bold", pad=5, fontsize=9.5)
axB.set_xlim(-0.03, 1.03)
axB.set_ylim(0, 1.55)   # headroom for legend at top
axB.set_yticks([0, 0.02, 0.1, 0.5, 1.0])
axB.set_yticklabels(["0", "0.02", "0.10", "0.50", "1.0"])
axB.axvline(0, color="#bdbdbd", lw=0.5, ls=":", zorder=0)
axB.grid(axis="y", color="#eeeeee", lw=0.4, which="major", zorder=0)
# Legend at the very top of the panel — the extra ylim headroom keeps it
# above every curve so the dash/dot samples never overlap the histograms.
axB.legend(frameon=False, loc="upper right", handlelength=2.0,
           handletextpad=0.5, borderaxespad=0.3, labelspacing=0.25,
           fontsize=7.5, bbox_to_anchor=(1.0, 1.0))

# Annotate the CSL spike. The annotation is placed in the empty band BETWEEN
# the histogram mass (which tops out at y~1.0 for the CSL spike) and the
# legend at the top of the panel. With ylim=(0, 2.4), the band y~[1.2, 1.7]
# is empty on both axes, so the text cannot overlap any curve.
csl_counts, _ = np.histogram(nn_csl, bins=BIN_EDGES)
csl_frac0 = csl_counts[0] / csl_counts.sum()
axB.annotate(
    "CSL-Daily: 99.66%\nof test at distance 0",
    xy=(0.025, csl_frac0 * 0.85),
    xytext=(0.30, 0.55),
    fontsize=7.0, color="#7b241c", fontweight="bold",
    ha="left", va="center",
    arrowprops=dict(arrowstyle="->", color="#7b241c", lw=0.9,
                    connectionstyle="arc3,rad=0.20",
                    shrinkA=2, shrinkB=2),
)


# =====================================================================
# Panel C — Delta-WER heatmap (method x dataset)
# =====================================================================
METHODS_C  = ["Dev-LM\nrescore", "BoW\n(dev-conditioned)",
              "BoW\n(train-conditioned)", "Cross-arch\nMBR"]
DATASETS_C = ["CSL", "PHX", "PHX-T"]

greedy = np.array([25.67, 19.62, 19.93])
delta = np.array([
    [14.20 - 25.67, 28.03 - 19.62, 32.45 - 19.93],   # Dev-LM
    [16.92 - 25.67, 25.04 - 19.62, 30.40 - 19.93],   # BoW dev-conditioned (20-seed)
    [37.49 - 25.67, 23.10 - 19.62, 30.47 - 19.93],   # BoW train-conditioned
    [25.90 - 25.67, 17.97 - 19.62, 18.57 - 19.93],   # Cross-arch MBR
])

vmax = max(abs(delta.min()), abs(delta.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
im = axC.imshow(delta, cmap="RdBu_r", norm=norm, aspect="auto")

axC.set_xticks(range(3))
axC.set_xticklabels(DATASETS_C, fontsize=8.2, fontweight="bold")
axC.set_yticks(range(4))
axC.set_yticklabels(METHODS_C, fontsize=7.5, linespacing=0.95)
axC.tick_params(axis="x", length=0, pad=3)
axC.tick_params(axis="y", length=0, pad=2)
axC.set_title("(c) $\\Delta$WER vs. greedy",
              loc="left", fontweight="bold", pad=5, fontsize=9.5)

for i in range(4):
    for j in range(3):
        v = delta[i, j]
        sign = "+" if v >= 0 else "$-$"
        col = "white" if abs(v) >= 6.0 else "#1c1c1c"
        axC.text(j, i, f"{sign}{abs(v):.1f}",
                 ha="center", va="center", fontsize=7.0,
                 color=col, fontweight="bold")

# Colorbar — attached to axC via axes_divider so it never collides with the
# heatmap cells or the row labels under tight_layout / bbox_inches="tight".
div = make_axes_locatable(axC)
cax = div.append_axes("right", size="6.5%", pad=0.10)
cb = fig.colorbar(im, cax=cax)
cb.set_label("$\\Delta$WER  (abs.)", fontsize=7.8, labelpad=2)
cb.ax.tick_params(labelsize=7.4, length=2)
cb.outline.set_linewidth(0.5)


# =====================================================================
# Top legend strip — global, for Panel A categories
# =====================================================================
legend_handles = [
    mpatches.Patch(facecolor=KIND_COLOR["pub"],     edgecolor="#222222",
                   label="Published CSLR backbone (no reranker)"),
    mpatches.Patch(facecolor=KIND_COLOR["devcond"], edgecolor="#222222",
                   hatch="//", label="Dev-conditioned probe (this paper, audit)"),
    mpatches.Patch(facecolor=KIND_COLOR["nodev"],   edgecolor="#222222",
                   label="No-dev-label control (this paper)"),
]
ax_leg.legend(
    handles=legend_handles, loc="center", ncol=3,
    frameon=False, fontsize=8.4, handlelength=1.6, handletextpad=0.5,
    columnspacing=2.0, borderaxespad=0,
)

out_pdf = OUT_DIR / "tripwire_figure.pdf"
out_png = OUT_DIR / "tripwire_figure.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=280, bbox_inches="tight")
plt.close(fig)
print("wrote", out_pdf)
