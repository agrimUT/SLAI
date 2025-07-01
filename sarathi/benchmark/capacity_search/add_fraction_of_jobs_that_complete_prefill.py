#!/usr/bin/env python3
"""
Add column  `fraction_sched_delay_prefill_lt1`
  = (# rows with request_scheduling_delay > 0 and prefill_e2e_time < 1)
    ------------------------------------------------------------------
    (# rows with request_scheduling_delay > 0)

to plots/<run_id>/capacity_curve.csv for every QPS row.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd


def read_flexible_csv(path: str) -> pd.DataFrame:
    """
    Try to read *either* comma- or tab-separated CSV, falling back automatically.
    Using sep=None with engine='python' lets pandas sniff the delimiter.
    """
    return pd.read_csv(path, sep=None, engine="python")  # auto-detects delimiter


def compute_fraction(seq_files, threshold=1.0):
    nonzero, good = 0, 0
    for f in seq_files:
        df = read_flexible_csv(f)
        missing = {"request_scheduling_delay", "prefill_e2e_time"} - set(df.columns)
        if missing:
            print(f"Warning: {f} missing {missing}, skipping")
            continue

        subset = df[df["request_scheduling_delay"] > 0]
        nonzero += len(subset)
        good += (subset["prefill_e2e_time"] < threshold).sum()

    return float("nan") if nonzero == 0 else good / nonzero


def main(base_dir, run_id, output=None):
    base = Path(base_dir)
    summary = base / "plots" / run_id / "capacity_curve.csv"
    df = pd.read_csv(summary)  # this one is standard comma-separated

    # compute per-QPS fractions
    fractions = {}
    for qps in df["qps"].unique():
        pattern = base / "runs" / run_id / str(qps) / "*" / "replica_*" / "sequence_metrics.csv"
        seq_files = glob.glob(str(pattern))
        fractions[qps] = compute_fraction(seq_files)

    df["fraction_sched_delay_prefill_lt1"] = df["qps"].map(fractions)

    out_path = summary if output is None else Path(output)
    df.to_csv(out_path, index=False)
    print(f"Wrote updated CSV to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a new file rather than overwriting capacity_curve.csv",
    )
    args = parser.parse_args()
    main(args.base_dir, args.run_id, args.output)
