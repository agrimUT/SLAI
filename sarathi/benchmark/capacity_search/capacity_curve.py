#!/usr/bin/env python3
"""
Generate capacity-curve data for every scheduler that declares a `capacity`
value in the YAML.

For each eligible JobConfig we run the benchmark at 10 QPS points between
0.1·capacity and capacity (inclusive).  After each run we record:

  • p99  decode_token_execution_plus_preemption_time
  • chosen-quantile (default P50) request_scheduling_delay
  • mean decode_token_execution_plus_preemption_time
  • mean batch_execution_time
  • mean batch_num_noncritical_decode_tokens
  • mean batch_num_time_critical_decode_tokens
  • scheduler name

Results are written to

  <output-dir>/plots/<job-hash>/capacity_curve.csv
"""

import argparse, glob, shlex, subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

import json, time
import ray
from sarathi.benchmark.capacity_search.ray_utils import ResourceManager

from sarathi.benchmark.capacity_search.config import (
    JobConfig,
    BenchmarkConfig,
    _get_hash,
)
from sarathi.logger import init_logger

logger = init_logger(__name__)
# --- PATCH 2: start Ray cluster client and one ResourceManager -------------
ray.init(ignore_reinit_error=True, log_to_driver=False)
_resource_mgr = ResourceManager.remote()
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", required=True,
                   help="Same YAML file used for capacity search.")
    p.add_argument("--output-dir", required=True,
                   help="Same --output-dir used for capacity search.")
    p.add_argument("--sched-delay-quantile", type=float, default=0.50,
                   help="Quantile for request_scheduling_delay (default 0.50).")
    p.add_argument("--time-limit", type=int, default=30,
                   help="Minutes per benchmark run (default 30).")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers (adapted from CapacitySearch)
# ----------------------------------------------------------------------
def _metric_csv(run_path: Path, metric: str) -> Path | None:
    matches = glob.glob(str(run_path / "**" / "plots" / f"{metric}.csv"),
                        recursive=True)
    return Path(matches[0]) if matches else None

def _get_seq_metric_csv(run_path: Path, metric: str) -> Path | None:
    matches = glob.glob(str(run_path / "**" /f"{metric}.csv"), recursive=True)
    return Path(matches[0]) if matches else None

# --- PATCH 4: launch benchmark with GPU mapping ---------------------------
def _launch_benchmark(cfg: BenchmarkConfig,
                      num_gpus: int,
                      log_file: Path) -> None:
    """
    Run sarathi.benchmark.main once.  Uses ResourceManager to pin the job to
    specific node/GPU IDs; frees them afterward.
    """
    if log_file.exists():
        return  # cached – metrics already there

    mapping = _acquire_mapping(num_gpus)
    mapping_cli = f"--replica_resource_mapping '{json.dumps(mapping)}'"

    cmd = f"python -m sarathi.benchmark.main {cfg.to_args()} {mapping_cli}"
    logger.debug(cmd)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(cmd + "\n")

    try:
        subprocess.run(
            shlex.split(cmd),
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
            check=True,
        )
    finally:
        # always release GPUs, even if the subprocess crashed
        ray.get(_resource_mgr.release_resources.remote(mapping))
# ---------------------------------------------------------------------------


def _acquire_mapping(num_gpus: int) -> dict:
    """
    Ask the ResourceManager for a free GPU bundle; block until one is free.
    """
    mapping = {}
    while not mapping:
        mapping = ray.get(
            _resource_mgr.get_replica_resource_mapping.remote(num_gpus)
        )
        if not mapping:
            time.sleep(0.2)
    return mapping

def get_quantile(csv_file: Path, metric: str, quantile: float) -> float:
    df = pd.read_csv(csv_file)
    return df.loc[df["cdf"] >= quantile, metric].iloc[0]

def get_mean(csv_file: Path, metric: str) -> float:
    df = pd.read_csv(csv_file)
    u = df["cdf"].to_numpy()                          # the quantile levels, from 0.0 → 1.0
    x = df[metric].to_numpy()                         # the corresponding values Q(u)
    mean_batch_exec_time = np.trapezoid(x, u)
    return mean_batch_exec_time

def get_mean_and_std(csv_file: Path, metric: str) -> Tuple[float, float]:
    df = pd.read_csv(csv_file)
    mean = df[metric].mean()
    std = df[metric].std()
    return mean, std
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = get_args()
    cfg_yaml = yaml.safe_load(open(args.config_path))
    jobs = JobConfig.generate_job_configs(cfg_yaml)

    runs_root = Path(args.output_dir) / "runs"
    plots_root = Path(args.output_dir) / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    q_label = f"p{int(args.sched_delay_quantile * 100)}"

    for job in jobs:
        cap = job.scheduler_config.capacity
        if cap is None:
            continue   # skip schedulers without declared capacity

        qps_grid = np.linspace(0.1 * cap, cap, 10).round(2).tolist()
        job_hash = _get_hash(job.get_key())
        logger.info(f"[{job_hash}] {job.get_human_readable_name()}  "
                    f"capacity={cap}  QPS grid={qps_grid}")

        rows = []
        for qps in qps_grid:
            bench_cfg = BenchmarkConfig(
                output_dir=args.output_dir,
                wandb_project=None,
                wandb_group=job.get_key(),
                wandb_sweep_id="",
                qps=qps,
                time_limit=args.time_limit,
                job_config=job,
            )

            run_dir = Path(bench_cfg.get_run_dir())
            _launch_benchmark(
                bench_cfg,
                num_gpus=job.get_num_gpus(),
                log_file=run_dir / "output.log",
            )
            sched_csv = _metric_csv(run_dir, "request_scheduling_delay") # TTFT 
            tbt_csv   = _metric_csv(run_dir, "decode_token_execution_plus_preemption_time") # TBT
            bet_csv = _metric_csv(run_dir, "batch_execution_time") # BET
            tcdt_csv = _metric_csv(run_dir, "batch_num_time_critical_decode_tokens") # TCDT
            ncdt_csv = _metric_csv(run_dir, "batch_num_noncritical_decode_tokens") # NCDT
            pft_csv = _metric_csv(run_dir, "batch_num_prefill_tokens") # PFT
            npp_csv = _metric_csv(run_dir, "batch_num_preempted_seq_prefill")
            npd_csv = _metric_csv(run_dir, "batch_num_preempted_seq_decode")
            sequence_csv = _get_seq_metric_csv(run_dir, "sequence_metrics") # SL
            # p99 decode‐token execution+preemption time
            p99_tbt = get_quantile(tbt_csv, "decode_token_execution_plus_preemption_time", 0.99) 
            # chosen scheduling‐delay quantile
            sched_delay_median = get_quantile(sched_csv, "request_scheduling_delay", args.sched_delay_quantile)
            mean_decode_tbt = get_mean(tbt_csv, "decode_token_execution_plus_preemption_time")
            mean_batch_exec_time = get_mean(bet_csv, "batch_execution_time")
            mean_noncritical_decode_tokens = get_mean(ncdt_csv, "batch_num_noncritical_decode_tokens")
            mean_timecritical_decode_tokens = get_mean(tcdt_csv, "batch_num_time_critical_decode_tokens")
            mean_prefill_tokens = get_mean(pft_csv, "batch_num_prefill_tokens")
            mean_preempted_prefill = get_mean(npp_csv, "batch_num_preempted_seq_prefill")
            mean_preempted_decode = get_mean(npd_csv, "batch_num_preempted_seq_decode")
            mean_pd_ratio, std_pd_ratio = get_mean_and_std(sequence_csv, "request_pd_ratio")
            mean_scheduling_delay, std_scheduling_delay = get_mean_and_std(sequence_csv, "request_scheduling_delay")

            rows.append(dict(
                scheduler=job.scheduler_config.name,
                qps=qps,
                p99_tbt_seconds=p99_tbt,
                mean_decode_tbt_seconds=mean_decode_tbt,
                mean_batch_exec_time_seconds=mean_batch_exec_time,
                mean_noncritical_decode_tokens=mean_noncritical_decode_tokens,
                mean_timecritical_decode_tokens=mean_timecritical_decode_tokens,
                mean_prefill_tokens=mean_prefill_tokens,
                mean_pd_ratio=mean_pd_ratio,
                std_pd_ratio=std_pd_ratio,
                mean_preempted_prefill=mean_preempted_prefill,
                mean_preempted_decode=mean_preempted_decode,
                **{f"sched_delay_{q_label}_seconds": sched_delay_median},
                mean_scheduling_delay=mean_scheduling_delay,
                std_scheduling_delay=std_scheduling_delay,
            ))

        # write one CSV per job
        out_dir = plots_root / job_hash
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "capacity_curve.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"Saved {out_csv}")

    logger.info("All jobs completed.")


if __name__ == "__main__":
    main()
