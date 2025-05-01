#!/usr/bin/env python3
"""
Generate capacity-curve data for every scheduler that declares a `capacity`
value in the YAML.

For each eligible JobConfig we run the benchmark at 10 QPS points between
0.1·capacity and capacity (inclusive).  After each run we record:

  • p99  decode_token_execution_plus_preemption_time
  • chosen-quantile (default P50) request_scheduling_delay
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

def _load_quantile(csv_file: Path, column: str, q: float) -> float:
    df = pd.read_csv(csv_file)
    return float(df[column].quantile(q))


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

        qps_grid = np.linspace(0.1 * cap, cap, 1).round(2).tolist()
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
            sched_csv = _metric_csv(run_dir, "request_scheduling_delay")
            tbt_csv = _metric_csv(run_dir,
                                  "decode_token_execution_plus_preemption_time")

            if sched_csv is None or tbt_csv is None:
                _launch_benchmark(
                    bench_cfg,
                    num_gpus=job.get_num_gpus(),        # ← PATCH 5: pass GPU count
                    log_file=run_dir / "output.log",
                )
                sched_csv = _metric_csv(run_dir, "request_scheduling_delay")
                tbt_csv = _metric_csv(
                    run_dir, "decode_token_execution_plus_preemption_time")

            p99_tbt = _load_quantile(
                tbt_csv,
                "decode_token_execution_plus_preemption_time",
                0.99,
            )
            sched_delay = _load_quantile(
                sched_csv,
                "request_scheduling_delay",
                args.sched_delay_quantile,
            )

            rows.append(dict(
                scheduler=job.scheduler_config.name,
                qps=qps,
                p99_tbt_seconds=p99_tbt,
                **{f"sched_delay_{q_label}_seconds": sched_delay},
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
