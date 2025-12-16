"""
Regression analysis visualization for logistic probe results.

This script creates regression plots showing how probe performance
changes across layer depths, with regression lines and confidence intervals.

Usage:
    python visualize_regression.py \
        --probe_results ./outputs/probe_results_deepseek-math-7b-instruct.json \
                        ./outputs/probe_results_deepseek-math-7b-rl.json \
        --output_dir ./figures
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import get_model_short_name, load_json


def plot_layer_performance_regression(results_list: List[dict], output_dir: Path):
    """
    Plots accuracy vs layer depth with regression lines.

    Args:
        results_list: List of probe results.
        output_dir: Target directory for plots.
    """
    # Set seaborn style for clean appearance
    sns.set_style("whitegrid")

    # Create figure with single subplot for Accuracy only - INCREASED SIZE for better visibility
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Define custom colors according to model type:
    # DeepSeek: dark red and red, Qwen: dark blue and blue
    # If only one of each given, use dark for RL, regular for Instruct/Base, etc.
    # Use Seaborn's 'deep' palette and select visually distinct, high-contrast colors
    # Always use the same color for the same model type for consistency and contrast
    deep_colors = sns.color_palette("RdBu", 10)
    # Assign specific colors for up to 6 models using color theory (contrasting hues)
    color_map = {
        "deepseek-rl": deep_colors[0],  # Blue
        "deepseek-instruct": deep_colors[1],  # Light Blue
        "deepseek-base": deep_colors[2],
        "olmo3-base": deep_colors[6],
        "olmo3-sft": deep_colors[7],
        "olmo3-dpo": deep_colors[8],
        "olmo3-rlzero": deep_colors[9],
        "default": deep_colors[4],  # Gray for others
    }
    colors = []
    for results in results_list:
        model_name = results["model_name"].lower()
        if "deepseek" in model_name:
            if "rl" in model_name:
                colors.append(color_map["deepseek-rl"])
            elif "instruct" in model_name:
                colors.append(color_map["deepseek-instruct"])
            else:
                colors.append(color_map["deepseek-base"])
        elif "allenai" in model_name:
            if "Olmo-3-7B-RLZero-Math" in model_name:
                colors.append(color_map["olmo3-rlzero"])
            elif "Olmo-3-7B-Think" in model_name:
                colors.append(color_map["olmo3-rlvr"])
            elif "Olmo-3-7B-Think-DPO" in model_name:
                colors.append(color_map["olmo3-dpo"])
            elif "Olmo-3-7B-Think-SFT" in model_name:
                colors.append(color_map["olmo3-sft"])
            else:
                colors.append(color_map["olmo3-base"])
        else:
            # fallback to a gray if unrecognized model
            colors.append(color_map["default"])

    # Only show Accuracy (not F1 Score)
    metric = "accuracy"
    metric_name = "Accuracy"

    # Find the maximum layer number across all models (final layer)
    max_layer = max(
        [lr["layer"] for results in results_list for lr in results["layer_results"]]
    )

    # Plot each model
    for model_idx, results in enumerate(results_list):
        model_name = results["model_name"]
        model_short = get_model_short_name(model_name)

        # Extract data and filter out negative layers and use max layer from all models
        layers = np.array(
            [
                lr["layer"]
                for lr in results["layer_results"]
                if 0 <= lr["layer"] <= max_layer
            ]
        )
        train = np.array(
            [
                lr["test"][metric]
                for lr in results["layer_results"]
                if 0 <= lr["layer"] <= max_layer
            ]
        )

        # Plot regression with seaborn - INCREASED MARKER AND LINE SIZE
        sns.regplot(
            x=layers,
            y=train,
            ax=ax,
            color=colors[model_idx],
            label=model_short,
            scatter_kws={"alpha": 0.7, "s": 80},
            line_kws={"linewidth": 3},
            ci=95,
        )

        # Add dashed line for this model's last layer - INCREASED WIDTH
        model_max_layer = max([lr["layer"] for lr in results["layer_results"]])
        ax.axvline(
            x=model_max_layer,
            color=colors[model_idx],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    # Add dashed line at final layer - INCREASED WIDTH
    ax.axvline(
        x=max_layer,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Final Layer",
    )

    # Styling - INCREASED FONT SIZES
    ax.set_xlabel("Layer", fontsize=16, fontweight="bold")
    ax.set_ylabel(f"Test {metric_name}", fontsize=16, fontweight="bold")
    ax.set_title(f"Layer Depth vs {metric_name}", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.8)

    # Set x-axis limits and custom ticks to create gap effect
    ax.set_xlim(-1, max_layer + 2)

    # Create custom ticks with gap before final layer
    tick_positions = list(range(0, max_layer, 5)) + [max_layer - 1, max_layer]
    tick_positions.sort()
    ax.set_xticks(tick_positions)

    # Adjust spacing: manually shift final layer visually by manipulating the scale
    # Alternative: just increase the distance in the tick labels
    tick_labels = [
        str(int(t)) if t != max_layer else f"  {max_layer}" for t in tick_positions
    ]
    ax.set_xticklabels(tick_labels, fontsize=12)

    # Set y-axis limits - Start from 0.7 to better show differences
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    # Save figure - INCREASED DPI for higher resolution
    output_path = output_dir / "layer_performance_regression.png"
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """CLI entry point for generating regression plots."""
    parser = argparse.ArgumentParser(
        description="Create regression plots for layer depth vs probe performance"
    )
    parser.add_argument(
        "--probe_results",
        nargs="+",
        required=True,
        help="Paths to probe results JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print(f"\n{'=' * 80}")
    print("LAYER PERFORMANCE REGRESSION ANALYSIS")
    print(f"{'=' * 80}\n")

    results_list = []
    for path in args.probe_results:
        print(f"Loading results from {path}...")
        results = load_json(Path(path))
        results_list.append(results)
        model_short = get_model_short_name(results["model_name"])
        n_layers = results["n_layers"]
        print(f"  Model: {model_short} ({n_layers} layers)")

    print(f"\nLoaded {len(results_list)} model(s)")

    # Generate regression plot
    print("\nGenerating regression plots...")
    plot_layer_performance_regression(results_list, output_dir)

    print(f"\n{'=' * 80}")
    print(f"Regression analysis saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
