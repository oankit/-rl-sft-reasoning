# Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs SFT Finetuned Models


## Abstract

Large reasoning models trained via reinforcement learning (RL) substantially outperform their supervised counterparts on tasks requiring logic and mathematical reasoning, yet the mechanistic basis for these improvements remains unclear. We investigate this phenomenon through an integrated behavioral-mechanistic analysis of mathematical reasoning, asking: *what internal differences enable RL models' improved reasoning capabilities?*

## Authors

* **Antyabha Rahman** (University of New South Wales)
* **Akshaj Gurugubelli** (Algoverse AI Research)
* **Omar Ankit** (University of Waterloo)
* **Kevin Zhu** (Algoverse AI Research)
* **Aishwarya Balwani** (St. Jude Children's Research Hospital) - *Corresponding Author*

## Website

Visit our project website: [https://oankit.github.io/-rl-sft-reasoning/](https://oankit.github.io/-rl-sft-reasoning/)

## Prerequisites

*   **Python 3.12+**
*   **NVIDIA GPU** (Recommended for model inference and activation extraction)
*   **[uv](https://docs.astral.sh/uv/)** package manager

## Installation

The experiment code is located in the `experiment_code` directory. We use `uv` for fast and reliable dependency management.

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Set up the environment**:
    ```bash
    cd experiment_code
    uv sync
    ```

## Usage

We primarily use `uv run` to run python files in scripts as it uses the dependencies installed.

### 1. Layer-wise Linear Probes

This experiment investigates the linear separability of internal representations across model layers. All scripts are located in `experiment_code/layerwise_linear_probe`.

**Step 1: Data Generation**
Generate synthetic math questions for the probing task.
```bash
uv run experiment_code/layerwise_linear_probe/question_generate.py
```

**Step 2: Generate Completions**
Run models to generate answers for the synthetic questions.
```bash
bash experiment_code/layerwise_linear_probe/generate_completion_script.sh
```

**Step 3: Extract Activations**
Extract and save the internal activations (hidden states) of the models during inference.
```bash
bash experiment_code/layerwise_linear_probe/extract_activation_script.sh
```

**Step 4: Train Probes**
Train linear probes on the extracted activations to predict correct/incorrect reasoning steps.
```bash
# Balance the dataset first
uv run experiment_code/layerwise_linear_probe/balance_probe_data_flexible.py

# Train probes across model families
bash experiment_code/layerwise_linear_probe/train_families_per_question.sh
```

**Step 5: Visualize Results**
Generate plots comparing probe performance across layers and models.
```bash
bash experiment_code/layerwise_linear_probe/visualize_families_per_question.sh
```

### 2. Token Variability Analysis

This experiment analyzes the variability of output tokens to understand generation diversity. Code is in `experiment_code/token_variability_experiment`.

*   **Run Experiments**: Use the automation script to run variability analysis across all supported models.
    ```bash
    bash experiment_code/token_variability_experiment/token_var_script.sh
    ```
*   **Individual Scripts**: Specific scripts for models (e.g., DeepSeek, Olmo) are in folders like `deepseek_math_scripts/` and `olmo3_scripts/`.
*   **Visualizations**: Generate bar graphs of accuracy vs. token coefficient of variation.
    ```bash
    uv run experiment_code/token_variability_experiment/bargraph_accuracy_vs_token_cv.py
    ```

### 3. Layerwise Ablation

This implements systematic activation patching to investigate the criticality of each layer in DeepSeek-Math models for mathematical reasoning capabilities.

*   **Run Experiments**:
    ``` bash
    uv run experiment_code/layerwise_ablation/main.py
    ```

*   **Visualizations**:
    ```bash
    uv run experiment_code/layerwise_ablation/visualizaiton.py
    ```

### 4. Correction Rerun (Version 2)

Version 2 of the paper reruns the probing and token-count experiments with corrected code. Sections 1 and 2 above describe the version 1 runs. The layer-wise ablation experiment was not rerun.

The final results behind version 2 are in `experiment_code/correction_rerun/final_results`: the paper's tables, the probe summaries, and the figures. Raw completions and activations are not in this repository, so Step 4 needs the output of Steps 1 to 3. The `*_results` and `balanced_outputs` directories hold the version 1 results.

**Step 1: Generate on a GPU**
Follow [`experiment_code/correction_rerun/pod/README.md`](experiment_code/correction_rerun/pod/README.md). `run_all.sh full` runs token variability, probe completions, and activation extraction for all four models. `run_part2.sh` then extracts DeepSeek-Math-RL activations at the final "The answer is:". The guide ends with the results in `results/full` and `results/part2`.

**Step 2: Train the probes**
Run `probe_eval.py` on each extraction directory. The analysis expects these outputs, where `<template>` is the template ID and `<tag>` is the model tag from the directory name:

| Output prefix | Flags | Model tags |
|---|---|---|
| `probe_eval_bal/<template>_1000_<tag>_tol1_all_bal` | `--scale --balance` | `deepseek_instruct`, `olmo3_instruct` |
| `probe_eval_results/<template>_1000_<tag>_tol1_all` | `--scale` | `deepseek_instruct`, `olmo3_instruct` |
| `probe_eval_part2/<template>_1000_<tag>_tol1_all_bal` | `--scale --balance` | `deepseek_rl`, `olmo3_thinking_rlvr` |
| `probe_eval_part2/<template>_1000_<tag>_tol1_all` | `--scale` | `deepseek_rl`, `olmo3_thinking_rlvr` |

Use `results/part2` for `deepseek_rl` and `results/full/probe` for the other tags. For example:
```bash
mkdir -p probe_eval_bal probe_eval_results probe_eval_part2
python experiment_code/layerwise_linear_probe/probe_eval.py \
    results/full/probe/785_1000_deepseek_instruct_outputs deepseek-math-7b-instruct \
    probe_eval_bal/785_1000_deepseek_instruct_tol1_all_bal --scale --balance
```

**Step 3: Arrange the results**
The analysis scripts read `RERUN_RESULTS`, which defaults to `experiment_code/correction_rerun/results`. Move everything there:
```bash
R=experiment_code/correction_rerun/results
mkdir -p $R/pod_results/part2 $R/pod_results/think300/token_var
mv results/full $R/pod_results/full
mv results/part2 $R/pod_results/part2/part2
cp -r $R/pod_results/full/token_var/olmo3_thinking_rlvr_results $R/pod_results/think300/token_var/
mv probe_eval_bal probe_eval_results probe_eval_part2 $R/
```
The analysis reads Olmo-3-Think token data from `think300`, because the paper's run finished that model in a later job.

**Step 4: Recompute the paper's numbers and figures**
Use an environment with numpy, pandas, scipy, scikit-learn, joblib, matplotlib, and seaborn:
```bash
cd experiment_code/correction_rerun/analysis
python tokvar_rerun_analysis.py
python tokvar_paper_numbers.py
python answer_accuracy.py
python trivial_baseline.py
python probe_paper_numbers.py
python make_fig2_rerun.py
python make_probe_figs_rerun.py
```
Tables go to `analysis/out`, and figures go to `correction_rerun/figures_rerun`. `tokvar_paper_numbers.py` asserts the paper's DeepSeek-Math-RL values for the 20-40% bin, so a new run with different values stops there. The confidence bands in the probe figures are bootstrapped without a seed, so they change slightly between runs. The fitted lines and points do not.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{rahman2025reasoning,
  title={Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs SFT Finetuned Models},
  author={Rahman, Antyabha and Gurugubelli, Akshaj and Ankit, Omar and Zhu, Kevin and Balwani, Aishwarya},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or correspondence, please contact:
*   **Aishwarya Balwani**: aishwarya.balwani@stjude.org
*   **Antyabha Rahman**: antyabha.rahman@student.unsw.edu.au

## Affiliation

Work conducted with **Algoverse AI Research**.

---
© 2025 Algoverse AI Research. All rights reserved.
