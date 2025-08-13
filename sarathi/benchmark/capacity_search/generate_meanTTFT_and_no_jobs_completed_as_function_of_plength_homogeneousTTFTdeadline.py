#!/usr/bin/env python3
"""
For each QPS value found in the supplied run directories, generate

  1. mean prefill_e2e_time vs prefill length
  2. number of sequences completed in each prefill-length bucket

One line per scheme, with deterministic colour + marker so the palette
matches other plots.

Usage
-----
python gen_prefill_plots_multi.py --out_dir <OUT_DIR> <RUN1> <RUN2> ...
"""

import argparse, glob, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────
# 1. path → friendly label
# ───────────────────────────────────────────────────────────────
PATH_LABEL_OVERRIDES = {
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms/runs/339e2590":
        r"sarathi-serve ($\tau = 512,\ \alpha = 128$)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms/runs/1fa6391d":
        r"last-minute ($\tau = 512,\ \alpha = 128,\ \beta = 128,\ \delta = 10$)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms_max/runs/a63e0e4b":
        r"SLAI ($\tau = 512,\ \alpha = 128,\ \beta = 128,\ \delta = 10$)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms/runs/0388ed44":
        r"last minute ($\tau = 512,\ \alpha = 128,\ \beta = 128,\ \delta = 5$)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms_max_reverseslack/runs/a63e0e4b":
        r"SLAI (reverse slack)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms_max_shortestjobfirst/runs/a63e0e4b":
        r"SLAI (shortest)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms_max_bucketsorting/runs/a63e0e4b":
        r"SLAI (bucket sorting)",
    "/home/ab73456/sarathi-serve/capacity_curve_wbinary_search_mistral_7b_relaxed_htr_pe2e_delay_p5per_100ms_500ms_max_bucketsorting_prefilldeadline2/runs/a63e0e4b":
        r"SLAI (bucket sorting, prefill deadline = 2s)",
}

# ───────────────────────────────────────────────────────────────
# 2. deterministic palette (same as earlier scripts)
# ───────────────────────────────────────────────────────────────
marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*']
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

LABELS_IN_ORDER = list(PATH_LABEL_OVERRIDES.values())
_marker_iter = itertools.cycle(marker_cycle)
_color_iter  = itertools.cycle(default_colors)

LABEL2MARKER = {lbl: next(_marker_iter) for lbl in LABELS_IN_ORDER}
LABEL2COLOR  = {lbl: next(_color_iter)  for lbl in LABELS_IN_ORDER}

def marker_for(lbl: str):
    if lbl not in LABEL2MARKER:
        LABEL2MARKER[lbl] = next(_marker_iter)
    return LABEL2MARKER[lbl]          # ← RETURN!

def color_for(lbl: str):
    if lbl not in LABEL2COLOR:
        LABEL2COLOR[lbl] = next(_color_iter)
    return LABEL2COLOR[lbl]           # ← RETURN!

# ───────────────────────────────────────────────────────────────
# 3. helpers
# ───────────────────────────────────────────────────────────────
def make_label(run_dir: Path) -> str:
    return PATH_LABEL_OVERRIDES.get(str(run_dir.resolve()),
                                    run_dir.name)

def read_any_csv(p): return pd.read_csv(p, sep=None, engine="python")

def concat_metrics(qps_dir: Path):
    patt = qps_dir / "*" / "replica_*" / "sequence_metrics.csv"
    files = glob.glob(str(patt))
    if not files:
        raise FileNotFoundError(f"No sequence_metrics.csv under {qps_dir}")
    return pd.concat([read_any_csv(f) for f in files], ignore_index=True)

def bucket_stats(df, bucket=512, max_tokens=8192):
    need = {"request_num_prefill_tokens", "prefill_e2e_time"}
    if need - set(df.columns):
        raise KeyError("required columns missing")
    edges = np.arange(0, max_tokens + bucket, bucket)
    idx = np.digitize(df["request_num_prefill_tokens"], edges, right=False) - 1
    centres, means, counts = [], [], []
    for i in range(len(edges) - 1):
        sub = df[idx == i]
        if sub.empty: continue
        centres.append((edges[i] + edges[i+1]) / 2)
        means.append(sub["prefill_e2e_time"].mean())
        counts.append(len(sub))
    return np.array(centres), np.array(means), np.array(counts)

# ───────────────────────────────────────────────────────────────
# 4. collect all runs
# ───────────────────────────────────────────────────────────────
def collect_data(run_dirs):
    """
    Returns:
        qps -> list[(label, x, mean_latency, count)]
    """
    out = {}
    for rd in run_dirs:
        label = make_label(rd)
        for qps_dir in filter(Path.is_dir, rd.iterdir()):
            qps = qps_dir.name
            try:
                df = concat_metrics(qps_dir)
                x, y_mean, y_cnt = bucket_stats(df)
                if x.size:
                    out.setdefault(qps, []).append((label, x, y_mean, y_cnt))
            except Exception as e:
                print(f"Skip {rd} (QPS {qps}): {e}")
    return out

# ───────────────────────────────────────────────────────────────
# 5. plotting
# ───────────────────────────────────────────────────────────────
def plot_all(qps_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    def plot_with_markers(ax, x, y, *, color, marker, label,
                      ms=8, lw=1.4, mew=1):
        ax.plot(
            x, y,
            color=color,
            linewidth=lw,
            linestyle='-',
            marker=marker,
            markeredgecolor=color,
            markeredgewidth=mew,
            label=label,
        )

    for qps, series in qps_map.items():
        series.sort(key=lambda t: LABELS_IN_ORDER.index(t[0])
                                if t[0] in LABELS_IN_ORDER else t[0])

        # ── 5-a mean latency ─────────────────────────────────────
        fig, ax = plt.subplots()
        for lbl, x, y_mean, _ in series:
            plot_with_markers(ax, x, y_mean,
                              color=color_for(lbl),
                              marker=marker_for(lbl),
                              label=lbl)
        ax.set_xlabel("Prefill length (tokens)")
        ax.set_ylabel("Mean prefill_e2e_time (s)")
        ax.set_title(f"Prefill latency vs. length — QPS {qps}")
        ax.grid(True); ax.legend(); fig.tight_layout()
        out = out_dir / f"prefill_length_vs_prefill_e2e_time_qps{qps}.pdf"
        fig.savefig(out); plt.close(fig); print("Wrote", out)

        # ── 5-b bucket counts ───────────────────────────────────
        fig2, ax2 = plt.subplots()
        for lbl, x, _, y_cnt in series:
            plot_with_markers(ax2, x, y_cnt,
                              color=color_for(lbl),
                              marker=marker_for(lbl),
                              label=lbl)
        ax2.set_xlabel("Prefill length (tokens)")
        ax2.set_ylabel("Number of completed prefill jobs")
        ax2.set_title(f"Number of prefills completed — QPS {qps}")
        ax2.grid(True); ax2.legend(); fig2.tight_layout()
        out2 = out_dir / f"number_prefills_complete_qps{qps}.pdf"
        fig2.savefig(out2); plt.close(fig2); print("Wrote", out2)


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
