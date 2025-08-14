# SLAI

This repository contains the code used to generate the plots in our paper:  
**"Optimal Scheduling Algorithms for LLM Inference: Theory and Practice"**

## Setup

### CUDA Environment

We use the same setup as [Sarathi-Serve (OSDI branch)](https://github.com/microsoft/sarathi-serve/tree/osdi-sarathi-serve).

SLAI was tested with:
- **CUDA version**: 12.1  
- **GPU**: NVIDIA RTX ADA 6000

Please ensure your machine has a compatible CUDA environment.

### Clone the Repository

```bash
git clone https://github.com/agrimUT/SLAI.git
cd SLAI
```

### Set Up Mamba Environment

If you don’t already have Mamba, install it using:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
```

Then create a Python 3.10 environment:

```bash
mamba create -p ./env python=3.10
mamba activate ./env
```

### Install SLAI and Dependencies

Install the package and its dependencies:

```bash
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
```

## Reproducing Results

We provide scripts to reproduce key plots from the paper.

### 1. Reproduce Figure 5

- **Plot (a)**: *Median TTFT and Batch Execution Time vs. Token Budget*

```bash
bash ./scripts/shell_script_to_generate_medianTTFT_and_bexectime_vs_token_budget.sh
python ./scripts/plotting_medianTTFT_and_bexectime_vs_token_budget.py
```

- **Plot (b)**: *GPU Memory Utilization vs. Number of Active Requests*

```bash
bash ./scripts/shell_script_to_generate_gpu_mem_utl_vs_active_requests.sh
python ./scripts/plotting_gpu_mem_utl_vs_active_requests.py
```

- **Plot (c)**: *Batch Execution Time vs. Number of Decode Tokens*

```bash
bash ./scripts/shell_script_to_generate_bexectime_vs_no_of_decodes.sh
python ./scripts/plotting_bexectime_vs_no_of_decodes.py
```

### 2. Evaluating a Given Policy and Configuration

To evaluate a specific scheduler policy, model, and mixture of user tiers:

```bash
python -m sarathi.benchmark.capacity_search.generate_TTFT_TBT_for_different_schedulers   --config-path ./config_path_yml_files/mistral7b_relaxed.yml   --output-dir ./heterogeneous_TBT_p5per_100ms_500ms_mistral7b   --prefill-time-quantile 0.50   --hetero_tbt_prob 0.05   --hetero_strict_tbt 0.1   --hetero_relaxed_tbt 0.5
```

#### Explanation of Parameters

- `--config-path`: Path to the YAML file that defines experiment settings including model name, tokenizer, batch size, max tokens, and scheduling policy.
- `--output-dir`: Directory where the generated metrics and logs will be saved.
- `--prefill-time-quantile`: Controls the target quantile (e.g., median) for prefill latency. Used to set SLAI's prefill latency target.
- `--hetero_tbt_prob`: Probability that a request comes from a **paying user**.
- `--hetero_strict_tbt`: Token Budget Time (TBT) deadline for **paying users**, measured in seconds.
- `--hetero_relaxed_tbt`: TBT deadline for **free-tier users**, measured in seconds.

This script simulates inference traffic under SLO constraints and evaluates how different schedulers (like SLAI, Sarathi-serve, vLLM, etc.) perform under heterogeneous user mixes.

## Project Structure

```
SLAI/
├── config_path_yml_files/        # Configuration files for models and scheduler policies
├── data/                         # Trace files and generated results
├── sarathi/                      # Core benchmarking and scheduling logic
│   ├── benchmark/                # Evaluation drivers, capacity search, metrics collection
│   └── core/                     # Sequence, scheduler, and cache logic
├── scripts/                      # Shell + Python scripts to run simulations and generate plots
├── env/                          # Conda environment directory (not tracked in Git)
├── README.md                     # Project documentation
└── setup.py                      # Setup script to install SLAI as a package
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{bari2025optimalschedulingalgorithmsllm,
  title={Optimal Scheduling Algorithms for LLM Inference: Theory and Practice}, 
  author={Agrim Bari and Parikshit Hegde and Gustavo de Veciana},
  year={2025},
  eprint={2508.01002},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2508.01002},
}
```

## Acknowledgments

This project builds on the [Sarathi-Serve](https://github.com/microsoft/sarathi-serve/tree/main) codebase. Like Sarathi-Serve, SLAI is a research prototype and does not aim for full feature parity with open-source vLLM. We have retained only the essential features to allow for faster iteration in research.