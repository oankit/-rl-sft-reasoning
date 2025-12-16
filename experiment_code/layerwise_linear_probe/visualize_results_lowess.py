"""
Step 4 (LOWESS): Visualize probe results and compare models using LOWESS smoothing.

Usage:
    python visualize_results_lowess.py \
        --probe_results ./outputs/probe_results_deepseek-math-7b-rl.json \
                        ./outputs/probe_results_deepseek-math-7b-instruct.json \
        --output_dir ./figures
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PathCollection
from matplotlib.typing import ColorType

from config import DEFAULT_FIGURES_DIR
from utils import get_model_short_name, load_json


# Family-based color palettes: shades within a single hue per model family.
# Darker shades = more training (RL > instruct > base).

BLUE_SHADES = {
    # DeepSeek family (blue hues)
    "base": "#A8D5E5",      # lightest
    "instruct": "#0072B2",  # medium
    "rl": "#004466",        # darkest
}

ORANGE_SHADES = {
    # OLMo family (orange/amber hues)
    "base": "#FFD480",      # lightest
    "instruct": "#FFAB40",  # light-medium
    "think-sft": "#E69F00", # medium
    "think-dpo": "#D55E00", # medium-dark
    "think": "#BF4000",     # dark (also used for rlvr)
    "rlzero": "#8B2500",    # darkest
}

# Utility colors
BLACK = "#000000"
GREY = "#999999"


def get_model_color(model_name: str) -> ColorType:
    """Returns a color based on model family (DeepSeek=Blue, OLMo=Orange) and training stage."""
    name = model_name.lower()

    # DeepSeek family -> blue shades
    if "deepseek" in name:
        if "rl" in name:
            return BLUE_SHADES["rl"]
        if "instruct" in name:
            return BLUE_SHADES["instruct"]
        return BLUE_SHADES["base"]

    # OLMo family -> orange shades
    if "allenai" in name or "olmo" in name:
        if "rlzero" in name:
            return ORANGE_SHADES["rlzero"]
        if "think-dpo" in name:
            return ORANGE_SHADES["think-dpo"]
        if "think-sft" in name:
            return ORANGE_SHADES["think-sft"]
        if "rlvr" in name:
            return ORANGE_SHADES["think"]  # same dark shade as think
        if "think" in name:
            return ORANGE_SHADES["think"]
        if "instruct" in name:
            return ORANGE_SHADES["instruct"]
        return ORANGE_SHADES["base"]

    return GREY


def plot_layer_performance_lowess(results_list: List[dict], output_dir: Path):
    """
    Plots test accuracy vs layer depth using LOWESS smoothing.

    Args:
        results_list: List of probe results.
        output_dir: Target directory for plots.
    """
    # Set seaborn style for clean appearance
    sns.set_style("whitegrid")

    # Create figure with single subplot for Accuracy only - INCREASED SIZE for better visibility
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.autoscale(enable=True, axis='y')

    # Assign a discrete, colorblind-friendly palette (Okabe–Ito) so each model is easy to
    # tell apart (avoids the "all blues" issue and improves print/readability).
    colors = [get_model_color(results["model_name"]) for results in results_list]

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
        # using LOWESS smoothing
        _before_collections = len(ax.collections)
        _before_lines = len(ax.lines)
        sns.regplot(
            x=layers,
            y=train,
            ax=ax,
            color=colors[model_idx] if model_idx < len(colors) else None,
            label=model_short,
            scatter_kws={"alpha": 0.6, "s": 80},
            line_kws={"linewidth": 3, "zorder": 3},
            lowess=True,  # Enable LOWESS smoothing
        )
        
        # Adjust scatter alpha if needed (though scatter_kws handles it)
        # Note: LOWESS in regplot does not produce CI bands, so no need to adjust poly collections
        for col in ax.collections[_before_collections:]:
            if isinstance(col, PathCollection):
                col.set_alpha(0.6)
                col.set_zorder(2)
        for line in ax.lines[_before_lines:]:
            line.set_zorder(3)

        # Add dashed line for this model's last layer - INCREASED WIDTH
        model_max_layer = max([lr["layer"] for lr in results["layer_results"]])
        # Use same color if available
        c = colors[model_idx] if model_idx < len(colors) else None
        ax.axvline(x=model_max_layer, color=c, linestyle="--", linewidth=1.5, alpha=0.7)

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
    ax.set_title(f"Layer Depth vs {metric_name} (LOWESS)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.8)

    # Set x-axis limits and custom ticks to create gap effect
    ax.set_xlim(-1, max_layer + 2)

    # Create custom ticks with gap before final layer
    tick_positions = list(range(0, max_layer, 5)) + [max_layer - 1, max_layer]
    tick_positions = sorted(list(set(tick_positions)))  # remove duplicates and sort
    ax.set_xticks(tick_positions)

    # Adjust spacing: manually shift final layer visually by manipulating the scale
    # Alternative: just increase the distance in the tick labels
    tick_labels = [
        str(int(t)) if t != max_layer else f"  {max_layer}" for t in tick_positions
    ]
    ax.set_xticklabels(tick_labels, fontsize=12)

    # Set y-axis limits - Start from 0 to better show full range (or modify as needed)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=12)
    plt.autoscale(enable=True, axis='y')
    plt.tight_layout()

    # Save figure - INCREASED DPI for higher resolution
    output_path = output_dir / "layer_performance_lowess.png"
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_by_layer(results_list: List[dict], output_dir: Path):
    """Plots simple line chart of accuracy vs layer for multiple models."""
    plt.figure(figsize=(12, 6))
    plt.autoscale(enable=True, axis='y')

    for results in results_list:
        model_name = results["model_name"]
        model_short = get_model_short_name(model_name)
        color = get_model_color(model_name)

        layers = [lr["layer"] for lr in results["layer_results"]]
        test_acc = [lr["test"]["accuracy"] for lr in results["layer_results"]]

        plt.plot(
            layers,
            test_acc,
            marker="o",
            label=model_short,
            linewidth=2,
            markersize=4,
            color=color,
        )

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Probe Accuracy by Layer", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "accuracy_by_layer.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(results_list: List[dict], output_dir: Path):
    """Plots subplots for accuracy, precision, recall, and F1 score."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    plt.autoscale(enable=True, axis='y')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for results in results_list:
            model_name = results["model_name"]
            model_short = get_model_short_name(model_name)
            color = get_model_color(model_name)

            layers = [lr["layer"] for lr in results["layer_results"]]
            values = [lr["test"][metric] for lr in results["layer_results"]]

            ax.plot(
                layers,
                values,
                marker="o",
                label=model_short,
                linewidth=2,
                markersize=4,
                color=color,
            )

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f"{metric.capitalize()} by Layer", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.autoscale(enable=True, axis='y')
    output_path = output_dir / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_train_val_test_comparison(results: dict, output_dir: Path):
    """Plots train/val/test accuracy for a single model to diagnose overfitting."""
    model_name = results["model_name"]
    model_short = get_model_short_name(model_name)

    layers = [lr["layer"] for lr in results["layer_results"]]
    train_acc = [lr["train"]["accuracy"] for lr in results["layer_results"]]
    val_acc = [lr["val"]["accuracy"] for lr in results["layer_results"]]
    test_acc = [lr["test"]["accuracy"] for lr in results["layer_results"]]

    plt.figure(figsize=(12, 6))
    plt.autoscale(enable=True, axis='y')
    plt.plot(layers, train_acc, marker="o", label="Train", linewidth=2, markersize=4)
    plt.plot(layers, val_acc, marker="s", label="Validation", linewidth=2, markersize=4)
    plt.plot(layers, test_acc, marker="^", label="Test", linewidth=2, markersize=4)

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        f"Train/Val/Test Accuracy - {model_short}", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"train_val_test_{model_short}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_emergence_analysis(
    results_list: List[dict], output_dir: Path, threshold: float = 0.6
):
    """Identifies and plots the first layer where accuracy exceeds threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    emergence_layers = []
    model_names = []
    bar_colors = []

    for results in results_list:
        model_name = results["model_name"]
        model_short = get_model_short_name(model_name)
        model_names.append(model_short)

        layers = [lr["layer"] for lr in results["layer_results"]]
        test_acc = [lr["test"]["accuracy"] for lr in results["layer_results"]]

        # Find first layer where accuracy >= threshold
        emergence_layer = None
        for layer, acc in zip(layers, test_acc):
            if acc >= threshold:
                emergence_layer = layer
                break

        if emergence_layer is not None:
            emergence_layers.append(emergence_layer)
            bar_colors.append("steelblue")
        else:
            # If never reaches threshold, use -1 to indicate N/A
            emergence_layers.append(-1)
            bar_colors.append("lightcoral")

        # Plot with threshold line
        ax1.plot(
            layers,
            test_acc,
            marker="o",
            label=model_short,
            linewidth=2,
            markersize=4,
            color=get_model_color(model_name),
        )

    ax1.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("Information Emergence Analysis", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bar plot of emergence layers
    bars = ax2.bar(range(len(model_names)), emergence_layers, color=bar_colors, alpha=0.7)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel("First Layer Above Threshold", fontsize=12)
    ax2.set_title(
        f"Emergence Layer (threshold={threshold})", fontsize=14, fontweight="bold"
    )

    # Set y-axis limits to show actual layer numbers
    max_layer = max([l for l in emergence_layers if l >= 0], default=0)
    ax2.set_ylim(-2, max(max_layer + 2, 5))

    # Add text labels on bars
    for i, (bar, layer) in enumerate(zip(bars, emergence_layers)):
        if layer >= 0:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"L{int(layer)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            # Show "N/A" for models that never reach threshold
            ax2.text(
                i, 0, "N/A", ha="center", va="center", fontsize=12, fontweight="bold"
            )

    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.autoscale(enable=True, axis='y')
    output_path = output_dir / "emergence_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(results_list: List[dict]):
    """Prints performance summary stats (best layer, mean accuracy) for all models."""
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}\n")

    for results in results_list:
        model_name = results["model_name"]
        model_short = get_model_short_name(model_name)

        test_accuracies = [lr["test"]["accuracy"] for lr in results["layer_results"]]
        best_layer = np.argmax(test_accuracies)
        best_accuracy = test_accuracies[best_layer]
        mean_accuracy = np.mean(test_accuracies)
        std_accuracy = np.std(test_accuracies)

        print(f"Model: {model_short}")
        print(f"  Best Layer: {best_layer}")
        print(f"  Best Test Accuracy: {best_accuracy:.4f}")
        print(f"  Mean Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Layers: {len(test_accuracies)}")
        print(
            f"  Train/Val/Test: {results['n_train']}/{results['n_val']}/{results['n_test']}"
        )
        print()

    print(f"{'=' * 80}\n")


def visualize_results(probe_results_paths: List[str], output_dir: str):
    """
    Main orchestration function to generate all visualization plots (including LOWESS).
    
    Args:
        probe_results_paths: List of JSON result file paths.
        output_dir: Destination for generated images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    results_list = []
    for path in probe_results_paths:
        print(f"Loading results from {path}...")
        results = load_json(Path(path))
        results_list.append(results)

    print(f"\nLoaded {len(results_list)} model result(s)")

    # Print summary statistics
    print_summary_statistics(results_list)

    # Generate plots
    print("Generating visualizations...")

    # 0. LOWESS Analysis (New)
    plot_layer_performance_lowess(results_list, output_dir)

    # 1. Accuracy by layer (main plot)
    plot_accuracy_by_layer(results_list, output_dir)

    # 2. All metrics comparison
    plot_metrics_comparison(results_list, output_dir)

    # 3. Train/val/test for each model
    for results in results_list:
        plot_train_val_test_comparison(results, output_dir)

    # 4. Emergence analysis
    plot_emergence_analysis(results_list, output_dir, threshold=0.6)

    print(f"\n{'=' * 80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize probe results using LOWESS")
    parser.add_argument(
        "--probe_results",
        nargs="+",
        required=True,
        help="Paths to probe results JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Accuracy threshold for emergence analysis",
    )

    args = parser.parse_args()

    visualize_results(args.probe_results, args.output_dir)


if __name__ == "__main__":
    main()

