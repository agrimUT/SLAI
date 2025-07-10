#!/usr/bin/env python3
"""
For each QPS value found in the supplied run directories, generate

  • mean prefill_e2e_time (“TTFT”) vs prefill length
  • number of sequences completed in each prefill-length bucket

done separately for strict-TTFT and relaxed-TTFT requests
(using the 'is_strict_prefill' column).

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

# ───────────────────────────────────────────────────────────────
# 1. path → friendly label  (unchanged – add your overrides here)
# ───────────────────────────────────────────────────────────────
PATH_LABEL_OVERRIDES = {
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p50per_100ms_500ms_TTFT_prefactor_10_20/runs/339e2590": 
        r"sarathi-serve ($\tau = 512,\ \alpha = 128$)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p50per_100ms_500ms_TTFT_prefactor_10_20/runs/a63e0e4b":
        r"experimental (bucket sorting)", 
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p50per_100ms_500ms_TTFT_prefactor_10_20_shortest/runs/a63e0e4b":
        r"experimental (shortest)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p50per_100ms_500ms_TTFT_prefactor_10_20_slack/runs/a63e0e4b":
        r"experimental (slack)",
    "/home/ab73456/sarathi-serve/heterogeneous_TBT_p50per_100ms_500ms_TTFT_prefactor_10_20/runs/1fa6391d":
        r"last-minute", 
}

# ───────────────────────────────────────────────────────────────
# 2. deterministic palette – keep colours consistent
# ───────────────────────────────────────────────────────────────
_marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LABELS_IN_ORDER = list(PATH_LABEL_OVERRIDES.values())
_marker_iter = itertools.cycle(_marker_cycle)
_color_iter  = itertools.cycle(_default_colors)

LABEL2MARKER = {lbl: next(_marker_iter) for lbl in LABELS_IN_ORDER}
LABEL2COLOR  = {lbl: next(_color_iter)  for lbl in LABELS_IN_ORDER}
def marker_for(lbl):                # one marker per *scheme*
    if lbl not in LABEL2MARKER:
        LABEL2MARKER[lbl] = next(_marker_iter)
    return LABEL2MARKER[lbl]
def color_for(lbl):
    if lbl not in LABEL2COLOR:
        LABEL2COLOR[lbl] = next(_color_iter)
    return LABEL2COLOR[lbl]

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

def _bucketed(df: pd.DataFrame, bucket: int = 512, max_tokens: int = 8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centres, mean latency, count per bucket."""
    edges = np.arange(0, max_tokens + bucket, bucket)
    bidx  = np.digitize(df["request_num_prefill_tokens"], edges, right=False) - 1
    centres, means, counts = [], [], []
    for i in range(len(edges) - 1):
        sub = df[bidx == i]
        if sub.empty: continue
        centres.append((edges[i] + edges[i+1]) / 2)
        means.append(sub["prefill_e2e_time"].mean())
        counts.append(len(sub))
    return np.asarray(centres), np.asarray(means), np.asarray(counts)

def bucket_stats_by_flag(df: pd.DataFrame,
                         flag_val: int,
                         **kw):
    sub = df[df["is_strict_prefill"] == flag_val]
    if sub.empty:                     # keep shape predictable
        return np.array([]), np.array([]), np.array([])
    return _bucketed(sub, **kw)

# ───────────────────────────────────────────────────────────────
# 4. collect all runs
# ───────────────────────────────────────────────────────────────
def collect_data(run_dirs):
    """
    Returns:
        qps -> {'strict': list[(label,x,mean,count)],
                'relaxed': list[(label,x,mean,count)]}
    """
    out: dict[str,dict[str,list]] = {}
    for rd in run_dirs:
        label = make_label(rd)
        for qps_dir in filter(Path.is_dir, rd.iterdir()):
            qps = qps_dir.name
            try:
                df = concat_metrics(qps_dir)
                for name, flag in (("strict", 1), ("relaxed", 0)):
                    x, y_mean, y_cnt = bucket_stats_by_flag(df, flag)
                    if x.size:
                        out.setdefault(qps, {}).setdefault(name, [])\
                           .append((label, x, y_mean, y_cnt))
            except Exception as e:
                print(f"Skip {rd} (QPS {qps}): {e}")
    return out

# ───────────────────────────────────────────────────────────────
# 5. plotting
# ───────────────────────────────────────────────────────────────
def _plot_series(ax, x, y, *, c, m, filled=True, lbl=None):
    face = c if filled else "none"
    ax.plot(x, y, color=c, marker=m, markerfacecolor=face,
            markeredgecolor=c, label=lbl, linewidth=1.4, markersize=8)

def plot_all(qps_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for qps, flavour_map in qps_map.items():
        for flavour in ("strict", "relaxed"):
            if flavour not in flavour_map:        # nothing to plot
                continue
            series = flavour_map[flavour]
            # put known schemes first
            series.sort(key=lambda t: t[0])

            # ── mean TTFT plot ────────────────────────────────────
            fig, ax = plt.subplots()
            for lbl, x, y_mean, _ in series:
                _plot_series(ax, x, y_mean,
                             c=color_for(lbl),
                             m=marker_for(lbl),
                             lbl=lbl)
            ax.set_xlabel("Prefill length (tokens)")
            ax.set_ylabel("Mean TTFT (s)")
            ax.set_title(f"Mean TTFT vs prefill length ({flavour}) — QPS {qps}")
            ax.grid(True); ax.legend(); fig.tight_layout()
            pdf = out_dir / f"mean_ttft_vs_prefill_len_{flavour}_qps{qps}_p50per.pdf"
            fig.savefig(pdf); plt.close(fig); print("Wrote", pdf)

            # ── bucket counts plot ───────────────────────────────
            fig2, ax2 = plt.subplots()
            for lbl, x, _, y_cnt in series:
                _plot_series(ax2, x, y_cnt,
                             c=color_for(lbl),
                             m=marker_for(lbl),
                             lbl=lbl)
            ax2.set_xlabel("Prefill length (tokens)")
            ax2.set_ylabel("# completed jobs")
            ax2.set_title(f"Jobs completed per prefill length ({flavour}) — "
                          f"QPS {qps}")
            ax2.grid(True); ax2.legend(); fig2.tight_layout()
            pdf2 = out_dir / f"num_completed_vs_prefill_len_{flavour}_qps{qps}_p50per.pdf"
            fig2.savefig(pdf2); plt.close(fig2); print("Wrote", pdf2)

# ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory for PDFs")
    ap.add_argument("run_dirs", nargs="+", help="/runs/<run_id> directories")
    args = ap.parse_args()

    runs = [Path(p).expanduser() for p in args.run_dirs]
    qps_data = collect_data(runs)
    if not qps_data:
        print("No data found – nothing plotted."); return
    plot_all(qps_data, Path(args.out_dir).expanduser())

if __name__ == "__main__":
    main()
