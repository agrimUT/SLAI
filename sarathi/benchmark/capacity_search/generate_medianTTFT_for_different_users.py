#!/usr/bin/env python3
"""
Plots (per scheduler):

  • median TTFT as a function of request-rate (QPS)
    – paying users   : is_strict_prefill == 1
    – free-tier users: is_strict_prefill == 0

Each scheduler/run contributes two curves (strict, relaxed).  
You can pass multiple run directories to compare schedulers
(e.g. Sarathi-Serve, SLAI, vLLM) on the same axes.

Usage
-----
python gen_median_ttft_vs_qps.py --out_dir <OUT_DIR> <RUN1> <RUN2> ...
"""
from __future__ import annotations
import argparse, glob, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────── style tweaks ────────────────────────────
plt.rcParams.update({
    "font.size"        : 16,
    "axes.labelsize"   : 16,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "legend.fontsize"  : 14,
    "axes.titlesize"   : 16,
})

# ───────────────────────────────────────────────────────────────
# 1. path → friendly label  (override here if desired)
# ───────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────
# 2. deterministic palette – 1 colour/marker per scheduler
# ───────────────────────────────────────────────────────────────
_marker_cycle  = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MAX_QPS = 1.6            

LABELS_IN_ORDER = list(PATH_LABEL_OVERRIDES.values())
_marker_iter = itertools.cycle(_marker_cycle)
_color_iter  = itertools.cycle(_default_colors)

LABEL2MARKER = {lbl: next(_marker_iter) for lbl in LABELS_IN_ORDER}
LABEL2COLOR  = {lbl: next(_color_iter)  for lbl in LABELS_IN_ORDER}
def marker_for(lbl):   return LABEL2MARKER.setdefault(lbl, next(_marker_iter))
def color_for(lbl):    return LABEL2COLOR.setdefault(lbl, next(_color_iter))

# ───────────────────────────────────────────────────────────────
# 3. helpers
# ───────────────────────────────────────────────────────────────
def make_label(run_dir: Path) -> str:
    return PATH_LABEL_OVERRIDES.get(str(run_dir.resolve()), run_dir.name)

def read_any_csv(p): return pd.read_csv(p, sep=None, engine="python")

def concat_metrics(qps_dir: Path) -> pd.DataFrame:
    patt  = qps_dir / "*" / "replica_*" / "sequence_metrics.csv"
    files = glob.glob(str(patt))
    if not files:
        raise FileNotFoundError(f"No sequence_metrics.csv under {qps_dir}")
    return pd.concat([read_any_csv(f) for f in files], ignore_index=True)

# ───────────────────────────────────────────────────────────────
# 4. collect data
# ───────────────────────────────────────────────────────────────
def collect_data(run_dirs):
    """
    Returns:
        flag → list[(label, qps_array, median_ttft_array)]
        where flag = 1 (strict / paying) or 0 (relaxed / free-tier)
    """
    out: dict[int, list] = {0: [], 1: []}

    for rd in run_dirs:
        label       = make_label(rd)
        strict_map  = {}   # qps → median
        relaxed_map = {}

        for qps_dir in filter(Path.is_dir, rd.iterdir()):
            try:
                qps_val = float(qps_dir.name)
            except ValueError:
                continue  # skip non-numeric dirs
            if qps_val > MAX_QPS:
                continue
            try:
                df = concat_metrics(qps_dir)
            except Exception as e:
                print(f"Skip {rd} (QPS {qps_dir.name}): {e}")
                continue

            for flag, target_map in ((1, strict_map), (0, relaxed_map)):
                sub = df[df["is_strict_prefill"] == flag]
                if not sub.empty:
                    target_map.setdefault(qps_val, []).extend(
                        sub["prefill_e2e_time"].values)

        # collapse lists → median
        for flag, mp in ((1, strict_map), (0, relaxed_map)):
            if not mp:
                continue
            qps_sorted  = np.array(sorted(mp))
            medians     = np.array([np.median(mp[q]) for q in qps_sorted])
            out[flag].append((label, qps_sorted, medians))

    return out

# ───────────────────────────────────────────────────────────────
# 5. plotting
# ───────────────────────────────────────────────────────────────
def _plot(ax, x, y, *, lbl, col, m):
    ax.plot(x, y, label=lbl, color=col,
            marker=m, markerfacecolor=col,
            markeredgecolor=col, markersize=8, linewidth=1.4)

def plot_all(flag_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    FLAG2NAME = {1: "paying (strict-TTFT)", 0: "free-tier (relaxed-TTFT)"}
    for flag, series in flag_map.items():
        if not series:
            continue
        series.sort(key=lambda t: t[0])  # stable legend order

        fig, ax = plt.subplots()
        for lbl, x, y in series:
            _plot(ax, x, y, lbl=lbl,
                  col=color_for(lbl), m=marker_for(lbl))

        ax.set_xlabel("Requests per second ")
        if flag == 1:
            ax.set_ylabel("Median TTFT (s) - paying users")
        else:
            ax.set_ylabel("Median TTFT (s) - free-tier users")
        ax.set_ylim(0, 2)  # fixed y-axis for better comparison
        ax.grid(True); ax.legend(frameon=True, framealpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"median_ttft_vs_qps_flag_{flag}.pdf")
        plt.close(fig)

# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory for PDF output")
    ap.add_argument("run_dirs", nargs="+", help="/runs/<run_id> directories")
    args = ap.parse_args()

    runs       = [Path(p).expanduser() for p in args.run_dirs]
    flag_data  = collect_data(runs)
    if not any(flag_data.values()):
        print("No data found – nothing plotted."); return
    plot_all(flag_data, Path(args.out_dir).expanduser())

if __name__ == "__main__":
    main()
