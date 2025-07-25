#!/usr/bin/env python3
"""
Plots (per QPS, per scheduler):

  • mean TTFT (prefill_e2e_time) vs. prefill length
  • number of requests completed in each prefill-length bucket

separately for strict-TTFT and relaxed-TTFT requests
('is_strict_prefill' column).

Usage
-----
python gen_prefill_plots_hetero.py --out_dir <OUT_DIR> <RUN1> <RUN2> ...
"""
from __future__ import annotations
import argparse, glob, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────── style tweaks ────────────────────────────
plt.rcParams.update({
    "font.size"        : 16,   # base font
    "axes.labelsize"   : 16,
    "xtick.labelsize"  : 14,
    "ytick.labelsize"  : 14,
    "legend.fontsize"  : 14,
    "axes.titlesize"   : 16,
})

# ───────────────────────────────────────────────────────────────
# 1. path → friendly label  (override paths here)
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
}

# ───────────────────────────────────────────────────────────────
# 2. deterministic palette – one colour+marker per scheme
# ───────────────────────────────────────────────────────────────
_marker_cycle  = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

LABELS_IN_ORDER = list(PATH_LABEL_OVERRIDES.values())
_marker_iter = itertools.cycle(_marker_cycle)
_color_iter  = itertools.cycle(_default_colors)

LABEL2MARKER = {lbl: next(_marker_iter) for lbl in LABELS_IN_ORDER}
LABEL2COLOR  = {lbl: next(_color_iter)  for lbl in LABELS_IN_ORDER}
def marker_for(lbl):   # unique marker per label
    return LABEL2MARKER.setdefault(lbl, next(_marker_iter))
def color_for(lbl):
    return LABEL2COLOR.setdefault(lbl, next(_color_iter))

# ───────────────────────────────────────────────────────────────
# 3. helpers
# ───────────────────────────────────────────────────────────────
def make_label(run_dir: Path) -> str:
    return PATH_LABEL_OVERRIDES.get(str(run_dir.resolve()), run_dir.name)

def read_any_csv(p): return pd.read_csv(p, sep=None, engine="python")

def concat_metrics(qps_dir: Path) -> pd.DataFrame:
    patt = qps_dir / "*" / "replica_*" / "sequence_metrics.csv"
    files = glob.glob(str(patt))
    if not files:
        raise FileNotFoundError(f"No sequence_metrics.csv under {qps_dir}")
    return pd.concat([read_any_csv(f) for f in files], ignore_index=True)

def _bucketed(df: pd.DataFrame,
              bucket: int = 512,
              max_tokens: int = 8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges   = np.arange(0, max_tokens + bucket, bucket)
    bindex  = np.digitize(df["request_num_prefill_tokens"], edges, right=False) - 1
    centres, means, counts = [], [], []
    for i in range(len(edges) - 1):
        sub = df[bindex == i]
        if sub.empty:
            continue
        centres.append((edges[i] + edges[i+1]) / 2)
        means.append(sub["prefill_e2e_time"].mean())
        counts.append(len(sub))
    return np.asarray(centres), np.asarray(means), np.asarray(counts)

def bucket_stats_by_flag(df: pd.DataFrame, flag_val: int, **kw):
    sub = df[df["is_strict_prefill"] == flag_val]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])
    return _bucketed(sub, **kw)

# ───────────────────────────────────────────────────────────────
# 4. collect all runs
# ───────────────────────────────────────────────────────────────
def collect_data(run_dirs):
    """
    Returns:
        qps -> {'strict':  [(label,x,mean,count), ...],
                'relaxed': [(label,x,mean,count), ...]}
    """
    out: dict[str, dict[str, list]] = {}
    for rd in run_dirs:
        label = make_label(rd)
        for qps_dir in filter(Path.is_dir, rd.iterdir()):
            qps = qps_dir.name
            try:
                df = concat_metrics(qps_dir)
                for flavour, flag in (("strict", 1), ("relaxed", 0)):
                    x, y_mean, y_cnt = bucket_stats_by_flag(df, flag)
                    if x.size:
                        out.setdefault(qps, {}).setdefault(flavour, [])\
                           .append((label, x, y_mean, y_cnt))
            except Exception as e:
                print(f"Skip {rd} (QPS {qps}): {e}")
    return out

# ───────────────────────────────────────────────────────────────
# 5. plotting
# ───────────────────────────────────────────────────────────────
def _plot(ax, x, y, *, lbl, col, m):
    face = col
    ax.plot(x, y, label=lbl, color=col,
            marker=m, markerfacecolor=face,
            markeredgecolor=col, markersize=8, linewidth=1.4)

def plot_all(qps_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for qps, flavour_map in qps_map.items():
        # ────────────────────────────────────────────────────────────
        # 5-a / 5-b  EXISTING PER-FLAVOUR FIGURES (unchanged)
        # ────────────────────────────────────────────────────────────
        for flavour in ("strict", "relaxed"):
            if flavour not in flavour_map:
                continue
            series = flavour_map[flavour]
            series.sort(key=lambda t: t[0])          # stable order

            # mean-TTFT plot
            fig, ax = plt.subplots()
            for lbl, x, y_mean, _ in series:
                _plot(ax, x, y_mean, lbl=lbl,
                      col=color_for(lbl), m=marker_for(lbl))
            ax.set_xlabel("Prompt length (tokens)")
            ax.set_ylabel("Mean TTFT (s)")
            ax.grid(True);  ax.legend(frameon=True, framealpha=0.5)
            fig.tight_layout()
            fig.savefig(out_dir /
                        f"mean_ttft_vs_prefill_len_{flavour}_qps_{qps}_p5per.pdf")
            plt.close(fig)

            # completed-jobs plot (per flavour)
            fig2, ax2 = plt.subplots()
            for lbl, x, _, y_cnt in series:
                _plot(ax2, x, y_cnt, lbl=lbl,
                      col=color_for(lbl), m=marker_for(lbl))
            ax2.set_xlabel("Prompt length (tokens)")
            ax2.set_ylabel("Number of requests completed")
            ax2.grid(True);  ax2.legend(frameon=True, framealpha=0.5)
            fig2.tight_layout()
            fig2.savefig(out_dir /
                         f"num_completed_vs_prefill_len_{flavour}_qps_{qps}_p5per.pdf")
            plt.close(fig2)

        # ────────────────────────────────────────────────────────────
        # 5-c  NEW: STRICT + RELAXED COUNT COMBINED
        # ────────────────────────────────────────────────────────────
        if not {"strict", "relaxed"} <= flavour_map.keys():
            continue          # need both to build the combined curves

        # Helper:   flavour → {label → (x-array, y-cnt-array)}
        def _to_dict(series):
            return {lbl: (x, y_cnt) for lbl, x, _, y_cnt in series}

        strict_d   = _to_dict(flavour_map["strict"])
        relaxed_d  = _to_dict(flavour_map["relaxed"])

        combined_series = []
        for lbl in sorted(set(strict_d) | set(relaxed_d)):
            # merge counts bucket-wise
            bucket_counts = {}
            for x_arr, y_arr in (strict_d.get(lbl, ([], [])),
                                 relaxed_d.get(lbl, ([], []))):
                for x_i, cnt_i in zip(x_arr, y_arr):
                    bucket_counts[x_i] = bucket_counts.get(x_i, 0) + cnt_i

            # convert back to sorted arrays
            x_all   = np.array(sorted(bucket_counts))
            y_all   = np.array([bucket_counts[xi] for xi in x_all])
            combined_series.append((lbl, x_all, y_all))

        # plotting
        figC, axC = plt.subplots()
        for lbl, x, y_cnt in combined_series:
            _plot(axC, x, y_cnt, lbl=lbl,
                  col=color_for(lbl), m=marker_for(lbl))
        axC.set_xlabel("Prompt length (tokens)")
        axC.set_ylabel("Number of requests served")
        axC.grid(True);  
        # axC.legend(frameon=True, framealpha=0.5)
        figC.tight_layout()
        figC.savefig(out_dir /
                     f"num_completed_vs_prefill_len_all_qps_{qps}_p5per.pdf")
        plt.close(figC)


# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory for PDF output")
    ap.add_argument("run_dirs", nargs="+", help="/runs/<run_id> directories")
    args = ap.parse_args()

    runs = [Path(p).expanduser() for p in args.run_dirs]
    qps_data = collect_data(runs)
    if not qps_data:
        print("No data found – nothing plotted."); return
    plot_all(qps_data, Path(args.out_dir).expanduser())

if __name__ == "__main__":
    main()
