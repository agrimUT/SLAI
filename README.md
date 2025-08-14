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

### 2. Evaluate a Given Policy and Configuration

To evaluate a policy with a specific model and a mixture of paying and free-tier users:

```bash
python -m sarathi.benchmark.capacity_search.generate_TTFT_TBT_for_different_schedulers   --config-path ./config_path_yml_files/mistral7b_relaxed.yml   --output-dir ./heterogeneous_TBT_p5per_100ms_500ms_mistral7b_spf_slai   --prefill-time-quantile 0.50   --hetero_tbt_prob 0.05   --hetero_strict_tbt 0.1   --hetero_relaxed_tbt 0.5
```

#### Explanation of Parameters

- `--prefill-time-quantile`: The quantile of prefill delay we focus on (e.g., 0.50 for median prefill delay).
- `--hetero_tbt_prob`: Probability that a request originates from a **paying user**.
- `--hetero_strict_tbt`: Token Budget Time (TBT) deadline for **paying users**, in seconds.
- `--hetero_relaxed_tbt`: TBT deadline for **free-tier users**, in seconds.

These parameters allow us to simulate and evaluate SLAI under heterogeneous service-level agreements (SLAs).

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