#!/usr/bin/env python3
"""
capacity_discovery_and_curve_prefill.py – discover max QPS under a
prefill-E2E-time SLO, then generate a 10-point capacity curve.

Main change vs. the original version:
  • All logic that formerly used request_scheduling_delay now uses
    prefill_e2e_time.
"""
from __future__ import annotations
import argparse, glob, json, shlex, subprocess, time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np, pandas as pd, ray, yaml

from sarathi.benchmark.capacity_search.config import (
    JobConfig, BenchmarkConfig, _get_hash
)
from sarathi.benchmark.capacity_search.ray_utils import ResourceManager
from sarathi.logger import init_logger
logger = init_logger(__name__)

# ─────────────────────────────────── CLI ────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config-path',   required=True)
    p.add_argument('--output-dir',    required=True)

    # *** NEW:  prefll-time SLO flags ***
    p.add_argument('--prefill-time-threshold', type=float, default=5.0,
                   help='Numeric SLO (seconds) for prefill_e2e_time')
    p.add_argument('--prefill-time-quantile',  type=float, default=0.50,
                   help='Quantile 0–1 or -1 for mean')

    # unchanged flags
    p.add_argument('--min-search-granularity', type=float, default=2.5)
    p.add_argument('--max-iterations',         type=int,   default=20)
    p.add_argument('--time_limit',             type=int,   default=2100,
                   help='Per-benchmark wall-clock limit (s)')
    p.add_argument('--debug', action='store_true')

    # heterogenous-TBT options remain unchanged
    p.add_argument('--hetero_tbt_prob',    type=float, default=0.0)
    p.add_argument('--hetero_strict_tbt',  type=float, default=0.1)
    p.add_argument('--hetero_relaxed_tbt', type=float, default=0.5)
    p.add_argument('--tbt-slo-value',      type=float, default=0.2)
    p.add_argument('--ttft_prefactor_strict',  type=float, default=5.0)
    p.add_argument('--ttft_prefactor_relaxed', type=float, default=10.0)
    return p.parse_args()

# ───────────────────────────── helper utilities ────────────────────────────
def _metric_csv(run_dir: Path, metric: str) -> Path | None:
    m = glob.glob(str(run_dir / '**' / 'plots' / f'{metric}.csv'),
                  recursive=True)
    return Path(m[0]) if m else None

def _get_seq_metric_csv(run_dir: Path, metric: str) -> Path | None:
    m = glob.glob(str(run_dir / '**' / f'{metric}.csv'), recursive=True)
    return Path(m[0]) if m else None

def _q(csv: Path, col: str, q: float):
    d = pd.read_csv(csv)
    return d.loc[d['cdf'] >= q, col].iloc[0]

def _mean_int(csv: Path, col: str):
    d = pd.read_csv(csv)
    return np.trapezoid(d[col].to_numpy(), d['cdf'].to_numpy())

def _mean_std(csv: Path, col: str):
    d = pd.read_csv(csv)
    return d[col].mean(), d[col].std()

# ──────────────────────────────── Ray set-up ───────────────────────────────
ray.init(ignore_reinit_error=True, log_to_driver=False)
_rm = ResourceManager.remote()

def _acquire(num_gpus: int):
    mapping = {}
    while not mapping:
        mapping = ray.get(_rm.get_replica_resource_mapping.remote(num_gpus))
        if not mapping:
            time.sleep(0.2)
    return mapping

def _run_bench(cfg: BenchmarkConfig, gpus: int):
    log = Path(cfg.get_run_dir()) / 'output.log'
    if log.exists():                      # already ran
        return

    mapping = _acquire(gpus)
    cmd = (f"python -m sarathi.benchmark.main {cfg.to_args()} "
           f"--replica_resource_mapping '{json.dumps(mapping)}'")
    if cfg.time_limit:
        cmd += f" --time_limit {cfg.time_limit}"

    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        f.write(cmd + '\n')
    try:
        subprocess.run(shlex.split(cmd),
                       stdout=open(log, 'a'),
                       stderr=subprocess.STDOUT,
                       check=True)
    finally:
        ray.get(_rm.release_resources.remote(mapping))

# ───────────────────────── phase 1: capacity search ────────────────────────
def _prefill_time(csv: Path, args) -> float:
    """Return the statistic we compare to the prefill-time SLO."""
    if args.prefill_time_quantile == -1:
        return _mean_int(csv, "prefill_e2e_time")
    return _q(csv, "prefill_e2e_time", args.prefill_time_quantile)

def _bucket_slo(bucket: str, args) -> float:
    if bucket == "strict":
        return args.hetero_strict_tbt or args.tbt_slo_value
    if bucket == "relaxed":
        return args.hetero_relaxed_tbt or args.tbt_slo_value
    return args.tbt_slo_value

def _tbt_files(run_dir: Path, hetero: bool) -> Dict[str, Path | None]:
    if hetero:
        return {
            "strict":  _metric_csv(run_dir,
                                   'decode_token_execution_plus_preemption_time_strict'),
            "relaxed": _metric_csv(run_dir,
                                   'decode_token_execution_plus_preemption_time_relaxed'),
        }
    return {"overall": _metric_csv(run_dir,
                                   'decode_token_execution_plus_preemption_time')}

_BASE_TBT_COL = "decode_token_execution_plus_preemption_time"
def _tbt_column(bucket: str) -> str:
    return (f"{_BASE_TBT_COL}_{bucket}"
            if bucket in ("strict", "relaxed")
            else _BASE_TBT_COL)

def _p99_tbt(csv: Path, bucket: str) -> float:
    raw = _q(csv, _tbt_column(bucket), 0.99)
    return raw 

def _deadline_fractions(seq_csv: Path) -> tuple[float, float]:
    """
    Return (strict_fraction, relaxed_fraction)
      – strict_fraction  :=  fraction of strict-deadline requests that met their
                            own deadline.
      – relaxed_fraction :=  same for relaxed-deadline requests.
        NaN if the bucket had zero requests.
    """
    df = pd.read_csv(seq_csv)
    # Guard against missing columns in very old runs.
    for col in ("prefill_e2e_time",
                "prefill_e2e_deadline",
                "is_strict_prefill"):
        if col not in df.columns:
            return float("nan"), float("nan")

    def _bucket_frac(strict: bool) -> float:
        bucket = df[df["is_strict_prefill"] == (1 if strict else 0)]
        if bucket.empty:
            return float("nan")
        met = bucket[bucket["prefill_e2e_time"] <= bucket["prefill_e2e_deadline"]]
        return len(met) / len(bucket)

    return _bucket_frac(True), _bucket_frac(False)

# ───────────────────────── phase 2: capacity curve ─────────────────────────
def _metrics(run_dir: Path, args):
    files = {
        'prefill'      : _metric_csv(run_dir, 'prefill_e2e_time'),
        'tbt_overall'  : _metric_csv(run_dir,
                                     'decode_token_execution_plus_preemption_time'),
        'tbt_strict'   : _metric_csv(run_dir,
                                     'decode_token_execution_plus_preemption_time_strict'),
        'tbt_relaxed'  : _metric_csv(run_dir,
                                     'decode_token_execution_plus_preemption_time_relaxed'),
        'batch_execution_time'              : _metric_csv(run_dir,
                                                          'batch_execution_time'),
        'batch_num_time_critical_decode_tokens': _metric_csv(run_dir,
                                                             'batch_num_time_critical_decode_tokens'),
        'batch_num_noncritical_decode_tokens': _metric_csv(run_dir,
                                                           'batch_num_noncritical_decode_tokens'),
        'batch_num_prefill_tokens'          : _metric_csv(run_dir,
                                                          'batch_num_prefill_tokens'),
        'batch_num_preempted_seq_prefill'   : _metric_csv(run_dir,
                                                          'batch_num_preempted_seq_prefill'),
        'batch_num_preempted_seq_decode'    : _metric_csv(run_dir,
                                                          'batch_num_preempted_seq_decode'),
        'sequence_metrics'                  : _get_seq_metric_csv(run_dir,
                                                                  "sequence_metrics"),
    }
    if any(v is None for v in files.values()):
        raise RuntimeError('missing csv')
    (prefill,tbt_all,tbt_strict,tbt_relaxed,bet,tcdt,ncdt,pft,npp,npd,seq
     ) = files.values()

    mean_pd,std_pd   = _mean_std(seq, 'request_pd_ratio')
    mean_pref,std_pref = _mean_std(prefill, 'prefill_e2e_time')
    frac_strict, frac_relaxed = _deadline_fractions(seq)

    qlab = (f"p{int(args.prefill_time_quantile*100)}"
            if args.prefill_time_quantile >= 0 else "mean")

    return {
        'p99_prefill_e2e_time_seconds':
            _q(prefill, 'prefill_e2e_time', 0.99),
        f'prefill_e2e_time_{qlab}_seconds':
            (_prefill_time(prefill, args)
             if args.prefill_time_quantile >= 0
             else _mean_int(prefill, 'prefill_e2e_time')),
        'mean_prefill_e2e_time_seconds':
            _mean_int(prefill, 'prefill_e2e_time'),

        'p99_tbt_seconds'        : _p99_tbt(tbt_all, "overall"),
        'p99_tbt_seconds_strict' : None if tbt_strict  is None
                                  else _p99_tbt(tbt_strict, "strict"),
        'p99_tbt_seconds_relaxed': None if tbt_relaxed is None
                                  else _p99_tbt(tbt_relaxed, "relaxed"),
        'mean_decode_tbt_seconds_overall':
            _mean_int(tbt_all, 'decode_token_execution_plus_preemption_time'),
        'mean_decode_tbt_seconds_strict':
            None if tbt_strict  is None else
            _mean_int(tbt_strict, 'decode_token_execution_plus_preemption_time_strict'),
        'mean_decode_tbt_seconds_relaxed':
            None if tbt_relaxed is None else
            _mean_int(tbt_relaxed, 'decode_token_execution_plus_preemption_time_relaxed'),
        'mean_batch_exec_time_seconds':
            _mean_int(bet, 'batch_execution_time'),
        'mean_noncritical_decode_tokens':
            _mean_int(ncdt,'batch_num_noncritical_decode_tokens'),
        'mean_timecritical_decode_tokens':
            _mean_int(tcdt,'batch_num_time_critical_decode_tokens'),
        'mean_prefill_tokens':
            _mean_int(pft,'batch_num_prefill_tokens'),
        'mean_preempted_prefill':
            _mean_int(npp,'batch_num_preempted_seq_prefill'),
        'mean_preempted_decode':
            _mean_int(npd,'batch_num_preempted_seq_decode'),

        'mean_pd_ratio'         : mean_pd,
        'std_pd_ratio'          : std_pd,
        'mean_prefill_e2e_time' : mean_pref,
        'strict_prefill_deadline_met_fraction'  : frac_strict,
        'relaxed_prefill_deadline_met_fraction' : frac_relaxed,
    }

def curve(job: JobConfig, cap: float, args, plots_root: Path):
    rows: List[Dict[str,Any]] = []
    for qps in np.linspace(0.1*cap, cap, 10).round(2):
    #for qps in [1.6]: 
        cfg = BenchmarkConfig(
            output_dir      = args.output_dir,
            wandb_project   = None,
            wandb_group     = job.get_key(),
            wandb_sweep_id  = '',
            qps             = qps,
            time_limit      = args.time_limit,
            job_config      = job,
            hetero_tbt_prob = args.hetero_tbt_prob,
            hetero_strict_tbt=args.hetero_strict_tbt,
            hetero_relaxed_tbt=args.hetero_relaxed_tbt, 
            ttft_prefactor_strict = args.ttft_prefactor_strict,
            ttft_prefactor_relaxed= args.ttft_prefactor_relaxed,
        )
        _run_bench(cfg, job.get_num_gpus())
        try:
            m = _metrics(Path(cfg.get_run_dir()), args)
        except RuntimeError as e:
            logger.warning(e); continue
        rows.append({'scheduler': job.scheduler_config.name,
                     'qps': qps, **m})

    out_dir = plots_root / _get_hash(job.get_key())
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'capacity_curve.csv', index=False)
    logger.info(f'saved {out_dir}/capacity_curve.csv')

# ─────────────────────────────────── main ──────────────────────────────────
def main():
    args = get_args()
    yaml_cfg = yaml.safe_load(open(args.config_path))
    jobs = JobConfig.generate_job_configs(yaml_cfg)

    plots_root = Path(args.output_dir) / 'plots'
    plots_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        curve(job, job.start_qps, args, plots_root)

    logger.info('done')

if __name__ == '__main__':
    main()
    ray.shutdown()
