#!/usr/bin/env python3
"""
Plot-2: mean batch latency vs number of decodes (bar chart, no error bars).

Reads
    /home/ab73456/sarathi-serve/benchmark_output/plot2/summary.csv

Writes (into …/scripts)
    • plot2_grouped.csv      – mean & std in *milliseconds*
    • plot2_mean_latency.pdf – bar figure (no error bars)
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


# ------------------------------------------------------------------ paths
CSV_IN  = pathlib.Path("/home/ab73456/sarathi-serve/benchmark_output/plot3/summary.csv")
OUT_DIR = pathlib.Path("/home/ab73456/sarathi-serve/scripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_OUT = OUT_DIR / "plot3_gpu_utilization.pdf"

# ------------------------------------------------------------------ read & aggregate
df = pd.read_csv(CSV_IN)

plt.figure(figsize=(6, 4))
# Use categorical positions 0,1,2,… so every bar is equally spaced
x_pos = range(len(df))
plt.bar(
    x_pos,
    (df["batch_gpu_blocks_reserved"]/14733) * 100,  # Convert to percentage
    width=0.8,
    color="tab:blue",
    edgecolor="black",
    alpha=0.9,
)

# ----- axis & style -----
plt.xticks(x_pos, df["N_decode"].astype(str), fontweight="bold")
plt.xlabel("Number of active requests", fontweight="bold")
plt.ylabel("GPU memory utilization (%)", fontweight="bold")

plt.grid(axis="y", ls="--", alpha=.4)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=1200)
print("Figure saved to:", FIG_OUT)
plt.show()
