"""
Balance probe training data across multiple models.
Usage:
    python balance_probe_data_flexible.py \
        --completions_paths path1.json path2.json path3.json \
        --output_dir ./balanced_outputs \
        --current_model_name instruct
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set
from utils import (
    load_json,
    save_json,
)

def load_completions(completions_path: str) -> List[Dict]:
    """
    Loads completions from a JSON file.

    Args:
        completions_path: Path to completions JSON.
    Returns:
        List of completion dictionaries (e.g., with 'index', 'is_correct').
    """
    print(f"Loading completions from {completions_path}...")
    completions = load_json(Path(completions_path))
    print(f"Loaded {len(completions)} completions")
    return completions


def extract_model_identifier(completions_path: str) -> str:
    """
    Extracts model identifier from file path.

    Args:
        completions_path: Path to completions JSON.
    Returns:
        Model name (e.g. "deepseek-math-7b-rl").
    """
    path = Path(completions_path)
    filename = path.stem
    if "completions_" in filename:
        return filename.replace("completions_", "")
    return path.stem


def find_intersection_indices(completions_dict: Dict[str, List[Dict]]) -> Set[int]:
    """
    Finds common question indices across all models.

    Args:
        completions_dict: Map of model names to completions.
    Returns:
        Set of indices valid for all models.
    """
    print("\nFinding intersection of valid indices...")
    
    model_indices = {}
    for model_name, completions in completions_dict.items():
        valid_indices = set()
        for completion in completions:
            if completion.get('index') is not None:
                valid_indices.add(completion['index'])
        model_indices[model_name] = valid_indices
        print(f"{model_name}: {len(valid_indices)} valid indices")
    
    # Find intersection
    intersection = set.intersection(*model_indices.values())
    print(f"Intersection: {len(intersection)} common indices")
    
    return intersection


def analyze_correctness_by_model(completions_dict: Dict[str, List[Dict]], 
                                intersection_indices: Set[int]) -> Dict[str, Dict]:
    """
    Categorizes intersection indices by correctness for each model.

    Args:
        completions_dict: Map of model names to completions.
        intersection_indices: Common indices to analyze.
    Returns:
        Stats dict per model with 'correct_indices', 'incorrect_indices', counts, and totals.
    """
    print("\nAnalyzing correctness within intersection...")
    
    model_stats = {}
    
    for model_name, completions in completions_dict.items():
        # Create index to completion mapping
        completion_map = {comp['index']: comp for comp in completions}
        
        correct_indices = []
        incorrect_indices = []
        
        for idx in intersection_indices:
            if idx in completion_map:
                completion = completion_map[idx]
                if completion.get('is_correct', False):
                    correct_indices.append(idx)
                else:
                    incorrect_indices.append(idx)
        
        model_stats[model_name] = {
            'correct_indices': correct_indices,
            'incorrect_indices': incorrect_indices,
            'n_correct': len(correct_indices),
            'n_incorrect': len(incorrect_indices),
            'total': len(correct_indices) + len(incorrect_indices)
        }
        
        print(f"{model_name}: {len(correct_indices)} correct, {len(incorrect_indices)} incorrect")
    
    return model_stats


def get_intersection_samples(model_stats: Dict[str, Dict], 
                             intersection_indices: Set[int],
                             random_state: int = 42) -> Dict[str, List[int]]:
    """
    Returns all valid intersection samples without subsampling.
    
    Args:
        model_stats: Model correctness statistics.
        intersection_indices: Common indices set (unused).
        random_state: Seed (unused).
    Returns:
        Map of model names to sorted list of all valid indices.
    """
    print("\nPreparing intersection samples (no subsampling)...")
    
    intersection_indices_list = {}
    
    for model_name, stats in model_stats.items():
        # Use ALL correct and incorrect indices from intersection
        all_indices = stats['correct_indices'] + stats['incorrect_indices']
        intersection_indices_list[model_name] = sorted(all_indices)
        
        print(f"\n{model_name}:")
        print(f"  Correct: {len(stats['correct_indices'])}, Incorrect: {len(stats['incorrect_indices'])}")
        print(f"  Total samples: {len(intersection_indices_list[model_name])}")
        
        # Report class ratio
        ratio = len(stats['correct_indices']) / len(all_indices) * 100
        print(f"  Class ratio: {ratio:.1f}% correct / {100-ratio:.1f}% incorrect")
    
    return intersection_indices_list


def create_intersection_metadata(intersection_indices: Set[int],
                                 model_stats: Dict[str, Dict],
                                 sample_indices: Dict[str, List[int]]) -> Dict:
    """
    Generates summary metadata for the dataset.

    Args:
        intersection_indices: Common indices set.
        model_stats: Per-model correctness stats.
        sample_indices: Final selected indices per model.
    Returns:
        Metadata dict with intersection info and model statistics.
    """
    metadata = {
        "intersection_info": {
            "intersection_size": len(intersection_indices),
            "strategy": "intersection_only_no_subsampling",
            "note": "Class imbalance handled by class_weight='balanced' in training",
            "n_models": len(model_stats)
        },
        "model_stats": {}
    }
    
    for model_name in model_stats.keys():
        n_correct = model_stats[model_name]['n_correct']
        n_incorrect = model_stats[model_name]['n_incorrect']
        total = len(sample_indices[model_name])
        
        metadata["model_stats"][model_name] = {
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "total": total,
            "correct_ratio": round(n_correct / total * 100, 1) if total > 0 else 0,
            "incorrect_ratio": round(n_incorrect / total * 100, 1) if total > 0 else 0
        }
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Balance probe training data across multiple models")
    parser.add_argument("--completions_paths", type=str, nargs='+', required=True,
                       help="Paths to completion JSON files for all models")
    parser.add_argument("--output_dir", type=str, default="./balanced_outputs",
                       help="Output directory for balanced indices")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--current_model_name", type=str, default=None,
                       help="Name/identifier for the current model (optional, auto-detected if not provided)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading completions from all models...")
    completions_dict = {}
    for path in args.completions_paths:
        model_id = extract_model_identifier(path)
        completions_dict[model_id] = load_completions(path)
    
    print(f"\nLoaded {len(completions_dict)} models: {', '.join(completions_dict.keys())}")
    
    intersection_indices = find_intersection_indices(completions_dict)
    
    model_stats = analyze_correctness_by_model(completions_dict, intersection_indices)
    
    sample_indices = get_intersection_samples(model_stats, intersection_indices, args.random_state)
    
    metadata = create_intersection_metadata(intersection_indices, model_stats, sample_indices)
    
    print(f"\nSaving intersection indices to {output_dir}...")
    for model_name, indices in sample_indices.items():
        output_path = output_dir / f"balanced_indices_{model_name}.json"
        save_json(indices, output_path)
        print(f"Saved {len(indices)} indices for {model_name} to {output_path}")
    
    metadata_path = output_dir / "balanced_metadata.json"
    save_json(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")
    
    print(f"Number of models: {len(completions_dict)}")
    print(f"Intersection size: {len(intersection_indices)}")
    for model_name, stats in metadata['model_stats'].items():
        print(f"\n{model_name}:")
        print(f"  Total: {stats['total']} ({stats['correct_ratio']}% correct, {stats['incorrect_ratio']}% incorrect)")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nIndices files created:")
    for model_name in sample_indices.keys():
        print(f"  - balanced_indices_{model_name}.json")


if __name__ == "__main__":
    main()

