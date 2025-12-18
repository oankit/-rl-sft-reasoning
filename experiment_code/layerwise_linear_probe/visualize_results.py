"""
Step 4: Visualize probe results and compare models.

Usage:
    python visualize_results.py \
        --probe_results ./outputs/probe_results_deepseek-math-7b-rl.json \
                        ./outputs/probe_results_deepseek-math-7b-instruct.json \
        --output_dir ./figures
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.typing import ColorType
from config import DEFAULT_FIGURES_DIR
from utils import get_model_short_name, load_json
from pathlib import Path

# Family-based color palettes: shades within a single hue per model family.
# Darker shades = more training (RL > instruct > base)
BLUE_SHADES = {
    # "base": "#A8D5E5",      # lightest
    "instruct": "#0072B2",  # medium
    "rl": "#004466",        # darkest
}

ORANGE_SHADES = {
    # "base": "#FFD480",      # lightest
    "instruct": "#FFAB40",  # light-medium
    # "think-sft": "#E69F00", # medium
    # "think-dpo": "#D55E00", # medium-dark
    "think": "#BF4000",     # dark
    # "rlzero": "#8B2500",    # darkest
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
        # return BLUE_SHADES["base"]


    # OLMo family -> orange shades
    if "allenai" in name or "olmo" in name:
        # if "rlzero" in name:
        #     return ORANGE_SHADES["rlzero"]
        # if "think-dpo" in name:
        #     return ORANGE_SHADES["think-dpo"]
        # if "think-sft" in name:
        #     return ORANGE_SHADES["think-sft"]
        if "think" in name or "rlvr" in name:
            return ORANGE_SHADES["think"]
        if "instruct" in name:
            return ORANGE_SHADES["instruct"]
        # return ORANGE_SHADES["base"]

    return GREY


def plot_layer_performance_regression(results_list: List[dict], output_dir: Path):
    """
    Plots test accuracy vs layer depth with linear regression fit.

    Args:
        results_list: List of probe results.
        output_dir: Target directory for plots.
    """
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plt.autoscale(enable=True, axis='y')

    colors = [get_model_color(results["model_name"]) for results in results_list]

    metric = "accuracy"
    metric_name = "Accuracy"

    max_layer = max(
        [lr["layer"] for results in results_list for lr in results["layer_results"]]
    )

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

        _before_collections = len(ax.collections)
        _before_lines = len(ax.lines)
        sns.regplot(
            x=layers,
            y=train,
            ax=ax,
            color=colors[model_idx] if model_idx < len(colors) else None,
            label=model_short,
            scatter_kws={"alpha": 0.6, "s": 80},
            line_kws={"linewidth": 4, "zorder": 3},
            ci=95,
        )
        for col in ax.collections[_before_collections:]:
            if isinstance(col, PolyCollection):
                col.set_alpha(0.12)
                col.set_zorder(1)
            elif isinstance(col, PathCollection):
                col.set_alpha(0.6)
                col.set_zorder(2)
        for line in ax.lines[_before_lines:]:
            line.set_zorder(3)


        model_max_layer = max([lr["layer"] for lr in results["layer_results"]])
        # c = colors[model_idx] if model_idx < len(colors) else None
        ax.axvline(x=model_max_layer, color='BLACK', linestyle="--", linewidth=2.5, alpha=0.7)

    ax.axvline(
        x=max_layer,
        color="black",
        linestyle="--",
        linewidth=2.5,
        alpha=0.7,
        label="Final Layer",
    )

    ax.set_xlabel("Layer", fontsize=22, fontweight="bold")
    ax.set_ylabel(f"Test {metric_name}", fontsize=22, fontweight="bold")
    ax.set_title(f"Layer Depth vs {metric_name}", fontsize=24, fontweight="bold")
    ax.legend(fontsize=18, loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.8)

    ax.set_xlim(-1, max_layer + 2)

    tick_positions = list(range(0, max_layer, 5)) + [max_layer - 1, max_layer]
    tick_positions = sorted(list(set(tick_positions)))
    ax.set_xticks(tick_positions)

    tick_labels = [
        str(int(t)) if t != max_layer else f"  {max_layer}" for t in tick_positions
    ]
    ax.set_xticklabels(tick_labels, fontsize=16)

    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=18)
    plt.autoscale(enable=True, axis='y')
    plt.tight_layout()

    output_path = output_dir / "layer_performance_regression.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
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
    Main orchestration function to generate all visualization plots.

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
    plot_layer_performance_regression(results_list, output_dir)
    plot_accuracy_by_layer(results_list, output_dir)
    plot_metrics_comparison(results_list, output_dir)

    # Train/val/test for each model
    for results in results_list:
        plot_train_val_test_comparison(results, output_dir)

    # Emergence analysis
    plot_emergence_analysis(results_list, output_dir, threshold=0.6)

    print(f"All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize probe results")
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
