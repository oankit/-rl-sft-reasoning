#!/bin/bash

# Script to train probes based on model family per question
QUESTIONS=(270 584 785 884 922 942)

BASE_DIR=$(pwd)

for QID in "${QUESTIONS[@]}"; do
    echo "================================================================================"
    echo "Processing Question ID: $QID"
    echo "================================================================================"

    echo "Processing DeepSeek family..."
    FAMILY="deepseek"
    BALANCED_OUT_DIR="${BASE_DIR}/balanced_outputs/${QID}/${FAMILY}"
    mkdir -p "$BALANCED_OUT_DIR"

    declare -a DS_PATHS=(
        # "${BASE_DIR}/${QID}_1000_deepseek_base_outputs/completions_deepseek-math-7b-base.json"
        "${BASE_DIR}/${QID}_1000_deepseek_instruct_outputs/completions_deepseek-math-7b-instruct.json"
        "${BASE_DIR}/${QID}_1000_deepseek_rl_outputs/completions_deepseek-math-7b-rl.json"
    )

    EXISTING_DS_PATHS=()
    for path in "${DS_PATHS[@]}"; do
        if [ -f "$path" ]; then
            EXISTING_DS_PATHS+=("$path")
        else
            echo "Note: File not found (skipping): $path"
        fi
    done

    if [ ${#EXISTING_DS_PATHS[@]} -gt 0 ]; then
        # Step 1: Balance Data
        echo "Balancing DeepSeek data..."
        python balance_probe_data_flexible.py \
            --completions_paths "${EXISTING_DS_PATHS[@]}" \
            --output_dir "$BALANCED_OUT_DIR"
        
        # Step 2: Train Probes
        echo "Training DeepSeek probes..."
        for completions_path in "${EXISTING_DS_PATHS[@]}"; do
             # Directory of the completions file
             SOURCE_DIR=$(dirname "$completions_path")
             FILENAME=$(basename "$completions_path")
             # Extract model identifier: remove "completions_" and ".json"
             MODEL_ID="${FILENAME#completions_}"
             MODEL_ID="${MODEL_ID%.json}"
             
             ACTIVATIONS_PATH="${SOURCE_DIR}/activations_${MODEL_ID}.npy"
             LABELS_PATH="${SOURCE_DIR}/labels_${MODEL_ID}.npy"
             METADATA_PATH="${SOURCE_DIR}/metadata_${MODEL_ID}.json"
             BALANCED_INDICES_PATH="${BALANCED_OUT_DIR}/balanced_indices_${MODEL_ID}.json"
             
             if [[ -f "$ACTIVATIONS_PATH" && -f "$LABELS_PATH" && -f "$METADATA_PATH" && -f "$BALANCED_INDICES_PATH" ]]; then
                 echo "Training probe for $MODEL_ID..."
                 python train_probes_balanced.py \
                     --activations_path "$ACTIVATIONS_PATH" \
                     --labels_path "$LABELS_PATH" \
                     --metadata_path "$METADATA_PATH" \
                     --balanced_indices_path "$BALANCED_INDICES_PATH" \
                     --output_dir "$BALANCED_OUT_DIR"
             else
                 echo "Missing required files for $MODEL_ID. Skipping training."
                 echo "Checked: $BALANCED_INDICES_PATH"
             fi
        done
    else
        echo "No DeepSeek files found for QID $QID."
    fi

    echo "Processing OLMo3 Family..."    
    FAMILY="olmo3"
    BALANCED_OUT_DIR="${BASE_DIR}/balanced_outputs/${QID}/${FAMILY}"
    mkdir -p "$BALANCED_OUT_DIR"

    declare -a OLMO_PATHS=(
        # "${BASE_DIR}/${QID}_1000_olmo3_base_outputs/completions_Olmo-3-1025-7B.json"
        "${BASE_DIR}/${QID}_1000_olmo3_instruct_outputs/completions_Olmo-3-7B-Instruct.json"
        # "${BASE_DIR}/${QID}_1000_olmo3_rl_zero_outputs/completions_Olmo-3-7B-RLZero-Math.json"
        "${BASE_DIR}/${QID}_1000_olmo3_thinking_dpo_outputs/completions_Olmo-3-7B-Think-DPO.json"
        "${BASE_DIR}/${QID}_1000_olmo3_thinking_sft_outputs/completions_Olmo-3-7B-Think-SFT.json"
        "${BASE_DIR}/${QID}_1000_olmo3_thinking_rlvr_outputs/completions_Olmo-3-7B-Think.json"   
    )

    # Filter for existing files
    EXISTING_OLMO_PATHS=()
    for path in "${OLMO_PATHS[@]}"; do
        if [ -f "$path" ]; then
            EXISTING_OLMO_PATHS+=("$path")
        else
            echo "Note: File not found (skipping): $path"
        fi
    done

    if [ ${#EXISTING_OLMO_PATHS[@]} -gt 0 ]; then
        # Step 1: Balance Data
        echo "Balancing OLMo3 data..."
        python balance_probe_data_flexible.py \
            --completions_paths "${EXISTING_OLMO_PATHS[@]}" \
            --output_dir "$BALANCED_OUT_DIR"
        
        # Step 2: Train Probes
        echo "Training OLMo3 probes..."
        for completions_path in "${EXISTING_OLMO_PATHS[@]}"; do
             SOURCE_DIR=$(dirname "$completions_path")
             FILENAME=$(basename "$completions_path")
             MODEL_ID="${FILENAME#completions_}"
             MODEL_ID="${MODEL_ID%.json}"
             
             ACTIVATIONS_PATH="${SOURCE_DIR}/activations_${MODEL_ID}.npy"
             LABELS_PATH="${SOURCE_DIR}/labels_${MODEL_ID}.npy"
             METADATA_PATH="${SOURCE_DIR}/metadata_${MODEL_ID}.json"
             BALANCED_INDICES_PATH="${BALANCED_OUT_DIR}/balanced_indices_${MODEL_ID}.json"
             
             if [[ -f "$ACTIVATIONS_PATH" && -f "$LABELS_PATH" && -f "$METADATA_PATH" && -f "$BALANCED_INDICES_PATH" ]]; then
                 echo "Training probe for $MODEL_ID..."
                 python train_probes_balanced.py \
                     --activations_path "$ACTIVATIONS_PATH" \
                     --labels_path "$LABELS_PATH" \
                     --metadata_path "$METADATA_PATH" \
                     --balanced_indices_path "$BALANCED_INDICES_PATH" \
                     --output_dir "$BALANCED_OUT_DIR"
             else
                 echo "Missing required files for $MODEL_ID. Skipping training."
                 echo "Checked: $BALANCED_INDICES_PATH"
             fi
        done
    else
         echo "No OLMo3 files found for QID $QID."
    fi

done

echo "All processing complete."

