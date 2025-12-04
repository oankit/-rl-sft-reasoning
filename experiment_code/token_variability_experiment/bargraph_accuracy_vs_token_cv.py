#!/usr/bin/env python3
"""
Script to create bar graph visualizations comparing mean accuracy vs output token_cv
for models, separated into Olmo3 and DeepSeek families.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import scipy.stats as stats
import glob
import os

def load_model_data(model_name: str, csv_directory: str) -> pd.DataFrame | None:
    """
    Load and process data for a specific model from a directory of CSV files.
    
    Aggregates data across multiple runs (CSV files), calculates mean accuracy,
    output token mean, output token standard deviation, and the coefficient of 
    variation (CV) for output tokens per question.

    Args:
        model_name (str): The name of the model being processed.
        csv_directory (str): The directory path containing the result CSV files.

    Returns:
        pd.DataFrame or None: A DataFrame containing aggregated metrics per question
        (index, mean_accuracy, output_token_mean, output_token_std, output_token_cv),
        or None if no valid data is found.
    """
    print(f"\nLoading data for {model_name}...")
    
    csv_files = glob.glob(f"{csv_directory}/*.csv") 
    
    if not csv_files:
        print(f"No CSV files found in {csv_directory}")
        return None
    
    print(f"Found {len(csv_files)} CSV files for {model_name}")
    
    csv_files.sort()
    all_data: list[pd.DataFrame] = []
    
    processed_files = 0
    skipped_files = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and df.shape[1] > 0 and len(df) >= 50:
                all_data.append(df[['index', 'is_correct', 'output_tokens']])
                processed_files += 1
            else:
                skipped_files += 1
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            skipped_files += 1
    
    if not all_data:
        print(f"No valid data files found in {csv_directory}")
        return None
        
    if skipped_files > 0:
        print(f"Skipped {skipped_files} empty or invalid files")
    
    combined = pd.concat(all_data, keys=range(len(all_data)))
    combined = combined.reset_index(level=0).rename(columns={'level_0': 'run'})
    
    print(f"Processed {processed_files} files")
    
    grouped = combined.groupby('index').agg({
        'is_correct': 'mean',
        'output_tokens': ['mean', 'std', 'count']
    }).reset_index()
    
    grouped.columns = ['index', 'mean_accuracy', 'output_token_mean', 'output_token_std', 'run_count']
    
    # Calculate Sample CV (Coefficient of Variation)
    grouped['output_token_cv'] = grouped['output_token_std'] / grouped['output_token_mean']
    
    # Apply bias correction for small sample sizes
    grouped['output_token_cv'] = grouped['output_token_cv'] * (1 + 1 / (4 * grouped['run_count']))
    
    # Handle edge cases: 0 mean implies 0 variation; infinite values set to 0.
    grouped.loc[grouped['output_token_mean'] == 0, 'output_token_cv'] = 0
    grouped['output_token_cv'] = grouped['output_token_cv'].replace([np.inf, -np.inf], 0)
    
    print(f"\nMetrics for {model_name}:")
    print(f"  Mean accuracy range: {grouped['mean_accuracy'].min():.2f} - {grouped['mean_accuracy'].max():.2f}")
    print(f"  Output token CV range: {grouped['output_token_cv'].min():.3f} - {grouped['output_token_cv'].max():.3f}")
    
    return grouped

def prepare_bar_data(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare data for bar graphs by binning accuracy and calculating average CV.
    
    Groups the data into accuracy bins (0-20%, 20-40%, etc.) and calculates
    the mean, standard deviation, count, and standard error of the Output Token CV
    for each bin.

    Args:
        data_dict (dict): A dictionary mapping model names to their corresponding 
                          metrics DataFrames.

    Returns:
        pd.DataFrame: A combined DataFrame containing bin statistics for all models.
    """
    results: list[pd.DataFrame] = []
    
    for model_name, df in data_dict.items():
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        df['accuracy_bin'] = pd.cut(df['mean_accuracy'], bins=bins, labels=labels, include_lowest=True)
        
        bin_stats = df.groupby('accuracy_bin', observed=True).agg({
            'output_token_cv': ['mean', 'std', 'count']
        }).reset_index()
        
        bin_stats.columns = ['accuracy_bin', 'avg_cv', 'std_of_cv', 'count']
        
        # Calculate Standard Error of the Mean (SEM)
        bin_stats['se_of_cv'] = bin_stats['std_of_cv'] / np.sqrt(bin_stats['count'])
        
        # Calculate 95% Confidence Interval (CI) using t-distribution.
        # We use t-distribution instead of normal approximation (1.96) because sample 
        # sizes per bin vary and are often small (N < 30).
        degrees_of_freedom = bin_stats['count'].values - 1
        
        t_crit = np.full(degrees_of_freedom.shape, np.nan)
        valid_mask = degrees_of_freedom > 0
        if np.any(valid_mask):
            # ppf(0.975) corresponds to two-tailed 95% confidence
            t_crit[valid_mask] = stats.t.ppf(0.975, degrees_of_freedom[valid_mask])
            
        bin_stats['ci_95'] = bin_stats['se_of_cv'] * t_crit
        
        bin_stats['model'] = model_name
        results.append(bin_stats)
    
    if not results:
        return pd.DataFrame(columns=['accuracy_bin', 'avg_cv', 'std_of_cv', 'count', 'se_of_cv', 'ci_95', 'model'])

    return pd.concat(results, ignore_index=True)

def plot_family_metrics(models_config: dict[str, str], data_dict: dict[str, pd.DataFrame], bar_data: pd.DataFrame, colors: dict[str, str], title: str, output_filename: str, y_limit: float, base_dir: str) -> Figure:
    """
    Helper function to plot metrics for a specific model family.

    Args:
        models_config (dict): Configuration dictionary for the models.
        data_dict (dict): Dictionary containing the raw data for the models.
        bar_data (pd.DataFrame): Prepared bar data with bins and stats.
        colors (dict): Color mapping for the models.
        title (str): Title for the plot.
        output_filename (str): Filename to save the plot.
        y_limit (float): Y-axis limit for consistency across plots.
        base_dir (str): Base directory for saving files.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    accuracy_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    x = np.arange(len(accuracy_bins))
    width = 0.15
    
    # Center the bars based on the number of models
    num_models = len(models_config)
    start_offset = -((num_models - 1) * width) / 2

    for i, model_name in enumerate(models_config.keys()):
        if model_name in data_dict:
            model_data = bar_data[bar_data['model'] == model_name]
            
            # Reindex to ensure all bins are present
            model_data = model_data.set_index('accuracy_bin').reindex(accuracy_bins).reset_index()
            model_data = model_data.fillna(0)
            
            offset = start_offset + (i * width)
            ax.bar(x + offset, model_data['avg_cv'], width, 
                   label=model_name, color=colors[model_name],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Plot error bars representing 95% Confidence Intervals
            ax.errorbar(x + offset, model_data['avg_cv'], 
                        yerr=model_data['ci_95'], fmt='none', 
                        ecolor='black', capsize=5, alpha=0.6, linewidth=1.5)
            
            # Add sample count (N) on top of each bar
            # We place the text just above the bar height (avg_cv) rather than the error bar
            # to ensure it remains visible even when error bars are huge/cut off.
            # We shift the text slightly to the left to avoid overlapping with the vertical error bar.
            for j, val in enumerate(model_data['avg_cv']):
                if val > 0:  # Only label bars that exist
                    count = int(model_data['count'].iloc[j])
                    # Position text slightly above the bar
                    y_pos = val + (y_limit * 0.02)
                    # Shift x left by 0.04 (approx 1/4 of bar width) to clear the error bar
                    ax.text(x[j] + offset - 0.04, y_pos, f"n={count}", 
                            ha='center', va='bottom', fontsize=8, rotation=90, color='black')

    ax.set_xlabel('Accuracy Range', fontsize=18, fontweight='bold')
    ax.set_ylabel('Average CV (Bias Corrected)', fontsize=18, fontweight='bold')
    ax.set_title(f"{title}\n(Error Bars: 95% CI of the Mean)", fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    
    ax.set_xticklabels(accuracy_bins, rotation=0, fontsize=16)
    
    ax.set_ylim(0, y_limit)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    
    output_path = os.path.join(base_dir, output_filename)
    fig.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    return fig

def main() -> None:
    plt.style.use('seaborn-v0_8-paper')
    base_dir = "."
    
    print("=" * 80)
    print("Loading data for all models...")
    print("=" * 80)
    
    deepseek_models = {
        'DeepSeek-Math-RL': f'{base_dir}/deepseek_math_rl_results',
        'DeepSeek-Math-Instruct': f'{base_dir}/deepseek_math_instruct_results',
        'DeepSeek-Math-Base': f'{base_dir}/deepseek_math_base_results'
    }

    olmo3_models = {
        "Olmo-3-7B-RL-Zero-Math": f'{base_dir}/olmo3_rl_zero_results',
        "Olmo-3-7B-Think": f'{base_dir}/olmo3_thinking_rlvr_results',
        "Olmo-3-7B-Think-SFT": f'{base_dir}/olmo3_thinking_sft_results',
        "Olmo-3-7B-Think-DPO": f'{base_dir}/olmo3_thinking_dpo_results',
        'Olmo-3-Base-7B': f'{base_dir}/olmo3_base_results',
    }

    deepseek_colors = {
        'DeepSeek-Math-RL': '#ff4900',
        'DeepSeek-Math-Instruct': '#fda102',
        "DeepSeek-Math-Base": '#fde003'
    }

    olmo3_colors = {
        "Olmo-3-7B-RL-Zero-Math": '#001433', # Very Dark Navy
        "Olmo-3-7B-Think": '#003D99',      # Navy Blue (Lightened)
        "Olmo-3-7B-Think-SFT": '#0052CC',  # Darker Blue
        "Olmo-3-7B-Think-DPO": '#3385FF',  # Vibrant Medium Blue
        'Olmo-3-Base-7B': '#B3D9FF',       # Very Light Blue
    }

    # Load data
    deepseek_data: dict[str, pd.DataFrame] = {}
    for model_name, csv_dir in deepseek_models.items():
        metrics = load_model_data(model_name, csv_dir)
        if metrics is not None:
            metrics['model'] = model_name
            deepseek_data[model_name] = metrics

    olmo3_data: dict[str, pd.DataFrame] = {}
    for model_name, csv_dir in olmo3_models.items():
        metrics = load_model_data(model_name, csv_dir)
        if metrics is not None:
            metrics['model'] = model_name
            olmo3_data[model_name] = metrics

    print("\n" + "=" * 80)
    print("Creating visualization...")
    print("=" * 80)

    deepseek_bar_data = prepare_bar_data(deepseek_data)
    olmo3_bar_data = prepare_bar_data(olmo3_data)

    # Calculate global Y limit
    # We calculate the limit based primarily on the average CV (bar heights) rather than 
    # the confidence intervals. This prevents single small-sample bins with huge 
    # error bars (e.g., N=2) from compressing the entire visualization.
    
    max_avg_deepseek = deepseek_bar_data['avg_cv'].max() if not deepseek_bar_data.empty else 0
    max_avg_olmo3 = olmo3_bar_data['avg_cv'].max() if not olmo3_bar_data.empty else 0
    global_max_avg = max(max_avg_deepseek, max_avg_olmo3)
    
    # Set limit to accommodate the tallest bar with some headroom (e.g., +60%)
    # This ensures readable bars even if some huge error bars extend off-chart.
    y_max_limit = global_max_avg * 1.6 if global_max_avg > 0 else 1.0

    # Plot DeepSeek
    plot_family_metrics(
        deepseek_models, deepseek_data, deepseek_bar_data, deepseek_colors,
        'DeepSeek-Math Models: Average Output Token CV by Accuracy Range',
        'bargraph_deepseek_accuracy_vs_token_cv.png',
        y_max_limit, base_dir
    )

    # Plot Olmo3
    plot_family_metrics(
        olmo3_models, olmo3_data, olmo3_bar_data, olmo3_colors,
        'Olmo3 Models: Average Output Token CV by Accuracy Range',
        'bargraph_olmo3_accuracy_vs_token_cv.png',
        y_max_limit, base_dir
    )
    
    # Save CSVs
    deepseek_bar_data.to_csv(f'{base_dir}/deepseek_bar_data_cv.csv', index=False)
    olmo3_bar_data.to_csv(f'{base_dir}/olmo3_bar_data_cv.csv', index=False)
    print("Bar data saved to CSV files")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics by Accuracy Range")
    print("=" * 80)

    print("\nDEEPSEEK MODELS:")
    print(deepseek_bar_data.to_string(index=False))

    print("\n" + "-" * 80)
    print("OLMO3 MODELS:")
    print(olmo3_bar_data.to_string(index=False))
    
    plt.show()

if __name__ == "__main__":
    main()
