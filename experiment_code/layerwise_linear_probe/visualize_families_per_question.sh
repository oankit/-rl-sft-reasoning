#!/bin/bash

# Script to visualize probe results per question and model family
# Iterates through QIDs, and runs visualize_results.py for each family.

# Questions to process
QUESTIONS=(270 584 785 884 922 942)

# Base directory (current dir)
BASE_DIR=$(pwd)

for QID in "${QUESTIONS[@]}"; do
    echo "================================================================================"
    echo "Generating visualizations for Question ID: $QID"
    echo "================================================================================"

    # 1. DeepSeek Models
    FAMILY="deepseek"
    BALANCED_OUT_DIR="${BASE_DIR}/balanced_outputs/${QID}/${FAMILY}"
    FIGURES_DIR="${BALANCED_OUT_DIR}/figures"
    
    if [ -d "$BALANCED_OUT_DIR" ]; then
        # Find all probe results files in this directory
        # Using find to get the list, then read into array
        # Note: we look for 'probe_results_balanced_*.json'
        
        # Use file expansion into array
        RESULTS_FILES=("$BALANCED_OUT_DIR"/probe_results_balanced_*.json)
        
        # Check if file expansion failed (if no files found, it keeps the pattern)
        if [ -e "${RESULTS_FILES[0]}" ]; then
            echo "Processing DeepSeek Family visualizations..."
            mkdir -p "$FIGURES_DIR"
            
            python visualize_results.py \
                --probe_results "${RESULTS_FILES[@]}" \
                --output_dir "$FIGURES_DIR"
        else
            echo "No probe results found for DeepSeek QID $QID (skipping)."
        fi
    else
        echo "Directory not found: $BALANCED_OUT_DIR"
    fi

    # 2. OLMo3 Models
    FAMILY="olmo3"
    BALANCED_OUT_DIR="${BASE_DIR}/balanced_outputs/${QID}/${FAMILY}"
    FIGURES_DIR="${BALANCED_OUT_DIR}/figures"
    
    if [ -d "$BALANCED_OUT_DIR" ]; then
        RESULTS_FILES=("$BALANCED_OUT_DIR"/probe_results_balanced_*.json)
        
        if [ -e "${RESULTS_FILES[0]}" ]; then
             echo "Processing OLMo3 Family visualizations..."
             mkdir -p "$FIGURES_DIR"
             
             python visualize_results.py \
                 --probe_results "${RESULTS_FILES[@]}" \
                 --output_dir "$FIGURES_DIR"
        else
             echo "No probe results found for OLMo3 QID $QID (skipping)."
        fi
    else
         echo "Directory not found: $BALANCED_OUT_DIR"
    fi

done

echo "All visualization generation complete."

