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

# ------------------------------------------------------------------ paths
CSV_IN  = pathlib.Path("/home/ab73456/sarathi-serve/benchmark_output/plot2/summary.csv")
OUT_DIR = pathlib.Path("/home/ab73456/sarathi-serve/scripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "plot2_grouped.csv"
FIG_OUT = OUT_DIR / "plot2_mean_latency.pdf"

# ------------------------------------------------------------------ read & aggregate
df = pd.read_csv(CSV_IN)

agg = (
    df.groupby("N_decode", as_index=False)["batch_exec_time_s"]
      .agg(mean_ms=lambda s: s.mean() * 1_000,
           std_ms =lambda s: s.std()  * 1_000)
      .rename(columns={"mean_ms": "mean_latency_ms",
                       "std_ms":  "std_latency_ms"})
      .sort_values("N_decode")
)

agg.to_csv(CSV_OUT, index=False)
print("Aggregated results saved to:", CSV_OUT)
print(agg.to_string(index=False))

# ------------------------------------------------------------------ plot (bars, no error bars)
plt.figure(figsize=(6, 4))
# Use categorical positions 0,1,2,… so every bar is equally spaced
x_pos = range(len(agg))
plt.bar(
    x_pos,
    agg["mean_latency_ms"],
    width=0.8,
    color="tab:blue",
    edgecolor="black",
    alpha=0.9,
)

# ----- axis & style -----
plt.xticks(x_pos, agg["N_decode"].astype(str), fontweight="bold")
plt.xlabel("Number of decodes in the batch", fontweight="bold")
plt.ylabel("Batch execution time (ms)", fontweight="bold")

plt.grid(axis="y", ls="--", alpha=.4)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=300)
print("Figure saved to:", FIG_OUT)
plt.show()
