#!/bin/bash

set -e

# Only if H100 series of GPUs are used
export VLLM_FLASH_ATTN_VERSION=3
HF_HOME="~/.cache/huggingface"
datasets=("270_1000" "584_1000" "785_1000" "884_1000" "922_1000" "942_1000")

# Set tensor and pipeline parallel size
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1


run_generation() {
    local model_name="$1"
    local dataset="$2"
    local output_dir="$3"
    shift 3
    
    local dataset_path="./q_data/${dataset}.csv"
    
    if [ ! -f "$dataset_path" ]; then
        echo "ERROR: Dataset not found: $dataset_path"
        return 1
    fi
    
    echo "Processing: $model_name on $dataset"
    echo "Output: $output_dir"
    echo ""
    if python3 generate_completions.py \
        --model_name "$model_name" \
        --dataset_path "$dataset_path" \
        --output_dir "$output_dir" \
        "$@"; then
        echo "Completed: $model_name on $dataset"
    else
        echo "Failed: $model_name on $dataset"
        return 1
    fi
}

echo "=== Deepseek-Math-7B-Base ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "deepseek-ai/deepseek-math-7b-base" \
        "$dataset" \
        "./${dataset}_deepseek_base_outputs" \
        --max_model_len 4096 \
        --max_new_tokens 4096 \
        --temperature 0.7 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Deepseek-Math-7B-Instruct ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "deepseek-ai/deepseek-math-7b-instruct" \
        "$dataset" \
        "./${dataset}_deepseek_instruct_outputs" \
        --max_model_len 4096 \
        --max_new_tokens 4096 \
        --temperature 0.7 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Deepseek-Math-7B-RL ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "deepseek-ai/deepseek-math-7b-rl" \
        "$dataset" \
        "./${dataset}_deepseek_rl_outputs" \
        --max_model_len 4096 \
        --max_new_tokens 4096 \
        --temperature 0.7 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Olmo-3-1025-7B ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "allenai/Olmo-3-1025-7B" \
        "$dataset" \
        "./${dataset}_olmo3_base_outputs" \
        --max_model_len 32768 \
        --max_new_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Olmo-3-7B-Instruct ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "allenai/Olmo-3-7B-Instruct" \
        "$dataset" \
        "./${dataset}_olmo3_instruct_outputs" \
        --max_model_len 32768 \
        --max_new_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Olmo-3-7B-RLZero-Math ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "allenai/Olmo-3-7B-RLZero-Math" \
        "$dataset" \
        "./${dataset}_olmo3_rl_zero_outputs" \
        --max_model_len 32768 \
        --max_new_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "=== Olmo-3-7B-Think ==="
for dataset in "${datasets[@]}"; do
    run_generation \
        "allenai/Olmo-3-7B-Think" \
        "$dataset" \
        "./${dataset}_olmo3_thinking_rlvr_outputs" \
        --max_model_len 32768 \
        --max_new_tokens 32768 \
        --temperature 0.6 \
        --top_p 0.95 \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE
done

echo "Generation completed successfully!"