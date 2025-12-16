#!/bin/bash

set -e

export HF_HOME="${HOME}/.cache/huggingface"

datasets=("270_1000" "584_1000" "785_1000" "884_1000" "922_1000" "942_1000")

get_completion_filename() {
    local model_name="$1"
    local short_name="${model_name##*/}"
    echo "completions_${short_name}.json"
}

run_extraction() {
    local model_name="$1"
    local dataset="$2"
    local output_dir="$3"
    local batch_size="$4"
    local completion_file=$(get_completion_filename "$model_name")
    local completions_path="${output_dir}/${completion_file}"
    
    if [ ! -f "$completions_path" ]; then
        echo "ERROR: Completions file not found: $completions_path"
        return 1
    fi
    
    echo "Processing: $model_name on $dataset"
    echo "Completions: $completions_path"
    echo "Output: $output_dir"
    
    if python3 extract_activations.py \
        --model_name "$model_name" \
        --completions_path "$completions_path" \
        --output_dir "$output_dir" \
        --batch_size "$batch_size"; then
        echo "Completed: $model_name on $dataset"
    else
        echo "Failed: $model_name on $dataset"
        return 1
    fi
}

echo "Starting activation extraction process..."

echo "=== DeepSeek-Math-7B-Base ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "deepseek-ai/deepseek-math-7b-base" \
        "$dataset" \
        "./${dataset}_deepseek_base_outputs" \
        1
done

echo "=== DeepSeek-Math-7B-Instruct ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "deepseek-ai/deepseek-math-7b-instruct" \
        "$dataset" \
        "./${dataset}_deepseek_instruct_outputs" \
        1
done

echo "=== DeepSeek-Math-7B-RL ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "deepseek-ai/deepseek-math-7b-rl" \
        "$dataset" \
        "./${dataset}_deepseek_rl_outputs" \
        1
done

echo "=== Olmo-3-1025-7B Base ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "allenai/Olmo-3-1025-7B" \
        "$dataset" \
        "./${dataset}_olmo3_base_outputs" \
        1
done

echo "=== Olmo-3-7B-Instruct ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "allenai/Olmo-3-7B-Instruct" \
        "$dataset" \
        "./${dataset}_olmo3_instruct_outputs" \
        1
done


echo "=== Olmo-3-7B-Think ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "allenai/Olmo-3-7B-Think" \
        "$dataset" \
        "./${dataset}_olmo3_thinking_rlvr_outputs" \
        1
done

echo "=== Olmo-3-7B-RLZero-Math ==="
for dataset in "${datasets[@]}"; do
    run_extraction \
        "allenai/Olmo-3-7B-RLZero-Math" \
        "$dataset" \
        "./${dataset}_olmo3_rl_zero_outputs" \
        1
done

echo "Activation extraction completed successfully!"
