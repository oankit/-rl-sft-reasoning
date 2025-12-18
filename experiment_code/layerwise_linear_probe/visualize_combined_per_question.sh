#!/bin/bash

# Script to visualize probe results per question with BOTH model families combined
# Iterates through QIDs, collects results from both deepseek and olmo3, and generates
# combined visualizations in a single diagram.

# Questions to process
QUESTIONS=(270 584 785 884 922 942)

# Base directory (current dir)
BASE_DIR=$(pwd)

for QID in "${QUESTIONS[@]}"; do
    echo "Generating COMBINED visualizations for Question ID: $QID"

    # Directories for each family
    DEEPSEEK_DIR="${BASE_DIR}/balanced_outputs/${QID}/deepseek"
    OLMO3_DIR="${BASE_DIR}/balanced_outputs/${QID}/olmo3"
    COMBINED_OUT_DIR="${BASE_DIR}/balanced_outputs/${QID}/combined_figures"

    # Collect all probe result files from both families
    ALL_RESULTS_FILES=()

    # Add DeepSeek results if directory exists
    if [ -d "$DEEPSEEK_DIR" ]; then
        DEEPSEEK_FILES=("$DEEPSEEK_DIR"/probe_results_balanced_*.json)
        if [ -e "${DEEPSEEK_FILES[0]}" ]; then
            ALL_RESULTS_FILES+=("${DEEPSEEK_FILES[@]}")
            echo "Found DeepSeek results: ${#DEEPSEEK_FILES[@]} file(s)"
        else
            echo "No DeepSeek probe results found for QID $QID"
        fi
    else
        echo "DeepSeek directory not found: $DEEPSEEK_DIR"
    fi

    # Add OLMo3 results if directory exists
    if [ -d "$OLMO3_DIR" ]; then
        OLMO3_FILES=("$OLMO3_DIR"/probe_results_balanced_*.json)
        if [ -e "${OLMO3_FILES[0]}" ]; then
            ALL_RESULTS_FILES+=("${OLMO3_FILES[@]}")
            echo "Found OLMo3 results: ${#OLMO3_FILES[@]} file(s)"
        else
            echo "No OLMo3 probe results found for QID $QID"
        fi
    else
        echo "OLMo3 directory not found: $OLMO3_DIR"
    fi

    # Generate combined visualization if we have results from at least one family
    if [ ${#ALL_RESULTS_FILES[@]} -gt 0 ]; then
        echo "Processing combined visualization with ${#ALL_RESULTS_FILES[@]} total file(s)..."
        mkdir -p "$COMBINED_OUT_DIR"

        python visualize_results.py \
            --probe_results "${ALL_RESULTS_FILES[@]}" \
            --output_dir "$COMBINED_OUT_DIR"

        echo "Combined figures saved to: $COMBINED_OUT_DIR"
    else
        echo "No probe results found for QID $QID from either family (skipping)."
    fi

    echo ""
done

echo "All combined visualization generation complete."
