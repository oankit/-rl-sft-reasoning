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

### 1. Layer-wise Linear Probes

This experiment investigates the linear separability of internal representations across model layers. All scripts are located in `experiment_code/layerwise_linear_probe`.

**Step 1: Data Generation**
Generate synthetic math questions for the probing task.
```bash
python experiment_code/layerwise_linear_probe/question_generate.py
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
python experiment_code/layerwise_linear_probe/balance_probe_data_flexible.py

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
    python experiment_code/token_variability_experiment/bargraph_accuracy_vs_token_cv.py
    ```

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
