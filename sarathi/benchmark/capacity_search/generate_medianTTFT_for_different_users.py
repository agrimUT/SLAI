#!/usr/bin/env python3
"""
Plots (per scheduler)

  • median TTFT vs QPS                     (separate PDFs for paying / free)
  • 99-th-pct TTFT vs QPS                  (separate PDFs for paying / free)
  • mean TTFT   vs QPS                     (separate PDFs for paying / free)

    – paying users   : is_strict_prefill == 1  (“strict” requests)
    – free-tier users: is_strict_prefill == 0  (“relaxed” requests)

Each run directory contributes two curves (paying, free-tier).  
Pass multiple run dirs to compare schedulers (Sarathi-Serve, SLAI, vLLM, …).

Example
-------
python gen_ttft_vs_qps.py --out_dir ~/plots </runs/*/339e2590> ...
"""
from __future__ import annotations
import argparse, glob, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ───────────────────────── style tweaks ────────────────────────────
plt.rcParams.update({
    "font.size"        : 16,
    "axes.labelsize"   : 16,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "legend.fontsize"  : 14,
    "axes.titlesize"   : 16,
})

# ─────────────────────── 1. path → pretty label ────────────────────
PATH_LABEL_OVERRIDES = {
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20/runs/339e2590":
        r"Sarathi-serve (FCFS)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20_shortest/runs/a63e0e4b":
        r"SLAI (SPF, fixed offset)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20/runs/1fa6391d":
        r"SLAI (FCFS, fixed offset)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20/runs/81067519":
        r"vLLM",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20_shortest/runs/339e2590":
        r"Sarathi-serve (SPF)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20_shortest_dynamic_offset_5_10_96/runs/a63e0e4b":
        r"SLAI (SPF, dynamic offset)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p5per_100ms_500ms_TTFT_prefactor_10_20_dynamic_offset/runs/a63e0e4b":
        r"SLAI (SPF with priority, dynamic offset)",
}

# ─────────────── 2. deterministic colours / markers ───────────────
_marker_cycle   = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MAX_QPS         = 1.6          # ignore runs above this load

LABELS_IN_ORDER = list(PATH_LABEL_OVERRIDES.values())
_marker_iter    = itertools.cycle(_marker_cycle)
_color_iter     = itertools.cycle(_default_colors)

LABEL2MARKER = {lbl: next(_marker_iter) for lbl in LABELS_IN_ORDER}
LABEL2COLOR  = {lbl: next(_color_iter)  for lbl in LABELS_IN_ORDER}
def marker_for(lbl): return LABEL2MARKER.setdefault(lbl, next(_marker_iter))
def color_for(lbl):  return LABEL2COLOR.setdefault(lbl, next(_color_iter))

# ──────────────────────────── helpers ──────────────────────────────
def make_label(run_dir: Path) -> str:
    return PATH_LABEL_OVERRIDES.get(str(run_dir.resolve()), run_dir.name)

def read_any_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=None, engine="python")

def concat_metrics(qps_dir: Path) -> pd.DataFrame:
    patt  = qps_dir / "*" / "replica_*" / "sequence_metrics.csv"
    files = glob.glob(str(patt))
    if not files:
        raise FileNotFoundError(f"No sequence_metrics.csv under {qps_dir}")
    return pd.concat([read_any_csv(f) for f in files], ignore_index=True)

# ───────────────────────── collect data ────────────────────────────
def collect_data(run_dirs):
    """
    Returns:
        flag → list[(label, qps[], median[], p99[], mean[])]
        flag = 1 (paying) or 0 (free-tier)
    """
    out: dict[int, list] = {0: [], 1: []}

    for rd in run_dirs:
        label       = make_label(rd)
        strict_map  = {}      # qps → list[ttft]
        relaxed_map = {}

        for qps_dir in filter(Path.is_dir, rd.iterdir()):
            try:
                qps_val = float(qps_dir.name)
            except ValueError:
                continue
            if qps_val > MAX_QPS:
                continue

            try:
                df = concat_metrics(qps_dir)
            except Exception as e:
                print(f"Skip {rd} (QPS {qps_dir.name}): {e}")
                continue

            for flag, tgt in ((1, strict_map), (0, relaxed_map)):
                sub = df[df["is_strict_prefill"] == flag]
                if not sub.empty:
                    tgt.setdefault(qps_val, []).extend(
                        sub["prefill_e2e_time"].values)

        # collapse lists → arrays
        for flag, mp in ((1, strict_map), (0, relaxed_map)):
            if not mp:
                continue
            qps_sorted = np.array(sorted(mp))
            medians = np.array([np.median   (mp[q]) for q in qps_sorted])
            p99s    = np.array([np.percentile(mp[q], 99) for q in qps_sorted])
            means   = np.array([np.mean     (mp[q]) for q in qps_sorted])

            out[flag].append((label, qps_sorted, medians, p99s, means))

    return out

# ──────────────────────────── plotting ─────────────────────────────
def _plot(ax, x, y, *, lbl, col, m):
    ax.plot(x, y, label=lbl, color=col,
            marker=m, markerfacecolor=col,
            markeredgecolor=col, markersize=8, linewidth=1.4)

def plot_all(flag_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for flag, series in flag_map.items():
        if not series:
            continue
        series.sort(key=lambda t: t[0])
        user_type = "paying users" if flag == 1 else "free-tier users"

        # 5-a median
        fig, ax = plt.subplots()
        for lbl, x, med, *_ in series:
            _plot(ax, x, med, lbl=lbl, col=color_for(lbl), m=marker_for(lbl))
        ax.set_xlabel("Requests per second")
        ax.set_ylabel(f"Median TTFT (s) – {user_type}")
        ax.set_ylim(0, 2)
        ax.grid(True); ax.legend(frameon=True, framealpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"median_ttft_vs_qps_flag_{flag}.pdf")
        plt.close(fig)

        # 5-b p99
        fig2, ax2 = plt.subplots()
        for lbl, x, _, p99, _ in series:
            _plot(ax2, x, p99, lbl=lbl, col=color_for(lbl), m=marker_for(lbl))
        ax2.set_xlabel("Requests per second")
        ax2.set_ylabel(f"99th-pct TTFT (s) – {user_type}")
        ax2.set_ylim(0, 60)           # adjust to your data
        ax2.grid(True); ax2.legend(frameon=True, framealpha=0.5)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"p99_ttft_vs_qps_flag_{flag}.pdf")
        plt.close(fig2)

        # 5-c mean
        fig3, ax3 = plt.subplots()
        for lbl, x, *rest in series:
            means = rest[-1]          # last element is mean array
            _plot(ax3, x, means, lbl=lbl, col=color_for(lbl), m=marker_for(lbl))
        ax3.set_xlabel("Requests per second")
        ax3.set_ylabel(f"Mean TTFT (s) – {user_type}")
        ax3.set_ylim(0, 4)            # tweak if needed
        ax3.grid(True); ax3.legend(frameon=True, framealpha=0.5)
        fig3.tight_layout()
        fig3.savefig(out_dir / f"mean_ttft_vs_qps_flag_{flag}.pdf")
        plt.close(fig3)

# ──────────────────────────── CLI entry ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory for PDF output")
    ap.add_argument("run_dirs", nargs="+", help="/runs/<run_id> directories")
    args = ap.parse_args()

    runs      = [Path(p).expanduser() for p in args.run_dirs]
    flag_data = collect_data(runs)

    if not any(flag_data.values()):
        print("No data found – nothing plotted.")
        return

    plot_all(flag_data, Path(args.out_dir).expanduser())

if __name__ == "__main__":
    main()
