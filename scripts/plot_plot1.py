#!/usr/bin/env python3
"""
Plot-1  •  Chunk-size sweep

Reads every run in /benchmark_output/plot1, aggregates
median TTFT & mean batch-time, writes plot1_results.csv,
and produces plot1_latency.pdf with bold axis labels.
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size"       : 14,   # base size for everything
    "axes.labelsize"  : 14,   # axis labels
    "axes.titlesize"  : 16,   # suptitle
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    "legend.fontsize" : 12,
})

ROOT = pathlib.Path("/home/ab73456/sarathi-serve/benchmark_output/plot1")
records = []

for chunk_dir in sorted(ROOT.glob("chunk_*")):
    chunk = int(chunk_dir.name.split("_")[1])
    if chunk > 1024:
        continue

    # newest timestamped run
    run_dirs = sorted(chunk_dir.glob("20*-*-*_*"), reverse=True)
    if not run_dirs:
        print(f"⚠️  no run found in {chunk_dir}")
        continue
    run_dir = run_dirs[0]
    rep0    = run_dir / "replica_0"

    batch_csv = rep0 / "batch_metrics.csv"
    seq_csv   = rep0 / "sequence_metrics.csv"
    if not (batch_csv.exists() and seq_csv.exists()):
        print(f"⚠️  missing CSVs for chunk {chunk}")
        continue

    batch_df = pd.read_csv(batch_csv)
    seq_df   = pd.read_csv(seq_csv)

    mean_batch  = batch_df["batch_execution_time"].mean()          # seconds
    median_ttft = seq_df["prefill_e2e_time"].median()              # seconds
    records.append((chunk, median_ttft * 1e3, mean_batch * 1e3))   # convert → ms

df = (
    pd.DataFrame(records, columns=["chunk", "ttft_ms", "batch_ms"])
      .sort_values("chunk")
      .reset_index(drop=True)
)

OUT_DIR = pathlib.Path("/home/ab73456/sarathi-serve/scripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "plot1_results.csv"
FIG_OUT = OUT_DIR / "plot1_latency.pdf"

df.to_csv(CSV_OUT, index=False)
print("\n=== Summary ===")
print(df.to_string(index=False))

# --------- plotting ----------
fig, ax1 = plt.subplots(figsize=(6.5, 4))

ax1.plot(df.chunk, df.ttft_ms, 'o-', color='tab:blue')
ax1.set_xlabel("Token budget", fontweight="bold")
ax1.set_ylabel("Median TTFT (ms)", fontweight="bold", color='tab:blue')
ax1.tick_params(axis='x', width=0)                       # no tick marks
ax1.tick_params(axis='y', labelcolor='tab:blue', width=0)

ax1.grid(ls='--', alpha=.4)

ax2 = ax1.twinx()
ax2.plot(df.chunk, df.batch_ms, 's--', color='tab:red')
ax2.set_ylabel("Mean batch\nexecution time (ms)",
               fontweight="bold", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red', width=0)

fig.tight_layout()
#fig.suptitle("Mistral-7B • Token Budget vs TTFT & Batch Time",
#             y=1.03, fontweight="bold", fontsize=16)

fig.savefig(FIG_OUT, dpi=1200)