#!/bin/bash

set -e

# Enabled Flash Attention 3 for Hopper GPUs (H200 is SM 9.0)
export VLLM_FLASH_ATTN_VERSION=3

export HF_HOME="$HOME/.cache/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"

CACHE_DIR="$HOME/.cache/huggingface/hub"

# Ensure the token file exists (create empty if needed)
mkdir -p "$HOME/.cache/huggingface"
touch "$HOME/.cache/huggingface/token"

# Function to run evaluation loop
run_evaluation() {
    local script_name=$1
    local result_dir=$2
    local csv_pattern=$3
    local runs=50

    echo "--- Starting Evaluation for $script_name ---"
    echo "Creating directory: $result_dir"
    mkdir -p "$result_dir"
    echo "Cleaning up old CSV files in $result_dir..."
    rm -f "$result_dir"/*.csv

    for i in $(seq 1 $runs); do
        echo "Run $i/$runs: $script_name"
        python3 "$script_name"
    done

    # Move results to the result directory
    echo "Moving results to $result_dir..."
    mv $csv_pattern "$result_dir/" 2>/dev/null || echo "No CSV files found matching $csv_pattern to move."
    
    echo "Clearing model files..."

    if [ -d "$CACHE_DIR" ]; then
        echo "Removing model cache at $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    fi

    echo "--- Completed Evaluation for $script_name ---"
    echo ""
}

# --- DeepSeek-Math-7B-Base Evaluation ---
run_evaluation "./deepseek_math_scripts/gsm8k_deepseek_math_base.py" "./deepseek_math_base_results" "evaluation_results_deepseek-ai_deepseek-math-7b-base_GSM8K_*.csv"

# --- DeepSeek-Math-7B-Instruct Evaluation ---
run_evaluation "./deepseek_math_scripts/gsm8k_deepseek_math_instruct.py" "./deepseek_math_instruct_results" "evaluation_results_deepseek-ai_deepseek-math-7b-instruct_GSM8K_*.csv"

# --- DeepSeek-Math-7B-RL Evaluation ---
run_evaluation "./deepseek_math_scripts/gsm8k_deepseek_math_rl.py" "./deepseek_math_rl_results" "evaluation_results_deepseek-ai_deepseek-math-7b-rl_GSM8K_*.csv"

# --- Olmo-3-1025-7B Base Evaluation ---
run_evaluation "./olmo3_scripts/gsm8k_olmo3_base.py" "./olmo3_base_results" "evaluation_results_allenai_Olmo-3-1025-7B_GSM8K_*.csv"

# --- Olmo-3-1025-7B Instruct Evaluation ---
run_evaluation "./olmo3_scripts/gsm8k_olmo3_instruct.py" "./olmo3_instruct_results" "evaluation_results_allenai_Olmo-3-1025-7B-Instruct_GSM8K_*.csv"

# --- Olmo-3-7B-Think Evaluation ---
run_evaluation "./olmo3_scripts/gsm8k_olmo3_thinking_rlvr.py" "./olmo3_thinking_rlvr_results" "evaluation_results_allenai_Olmo-3-7B-Think_GSM8K_*.csv"

# --- Olmo-3-7B-RLZero-Math Evaluation ---
run_evaluation "./olmo3_scripts/gsm8k_olmo3_rl_zero.py" "./olmo3_rl_zero_results" "evaluation_results_allenai_Olmo-3-7B-RLZero-Math_GSM8K_*.csv"

# Run the bargraph_accuracy_vs_token_cv.py script
python3 "./bargraph_accuracy_vs_token_cv.py"

echo "All evaluations finished."
