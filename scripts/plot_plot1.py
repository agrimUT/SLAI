#!/usr/bin/env python3
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/home/ab73456/sarathi-serve/benchmark_output/plot1")
records = []

for chunk_dir in sorted(ROOT.glob("chunk_*")):
    chunk = int(chunk_dir.name.split("_")[1])
    if chunk > 1024:
        continue 
    # there may be multiple timestamped runs; pick the newest one
    run_dirs = sorted(chunk_dir.glob("20*-*-*_*"), reverse=True)
    if not run_dirs:
        print(f"⚠️  no run found in {chunk_dir}")
        continue
    run_dir = run_dirs[0]                          # newest timestamped folder
    rep0    = run_dir / "replica_0"

    batch_csv = rep0 / "batch_metrics.csv"
    seq_csv   = rep0 / "sequence_metrics.csv"
    if not (batch_csv.exists() and seq_csv.exists()):
        print(f"⚠️  missing CSVs for chunk {chunk}")
        continue

    batch_df  = pd.read_csv(batch_csv)
    seq_df    = pd.read_csv(seq_csv)

    mean_batch  = batch_df["batch_execution_time"].mean()
    median_ttft = seq_df["prefill_e2e_time"].median()
    records.append((chunk, median_ttft, mean_batch))

df = (
    pd.DataFrame(records, columns=["chunk", "ttft", "batch"])
      .sort_values("chunk")
      .reset_index(drop=True)
)

df.to_csv("plot1_results.csv", index=False)
print("\n=== Summary ===")
print(df.to_string(index=False))

# --------- plotting ----------
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(df.chunk, df.ttft, 'o-', color='tab:blue', label='Median TTFT')
ax1.set_xlabel("Token Budget (chunk size)")
ax1.set_ylabel("Median TTFT (s)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(df.chunk, df.batch, 's--', color='tab:red', label='Mean Batch Time')
ax2.set_ylabel("Mean Batch Time (s)", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.grid(True, ls='--', alpha=.4)
fig.tight_layout()
fig.suptitle("Mistral-7B • Token Budget vs TTFT & Batch Time", y=1.05)

fig.savefig("plot1.pdf", dpi=1200)
plt.show()
