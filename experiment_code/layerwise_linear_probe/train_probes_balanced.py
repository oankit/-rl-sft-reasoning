"""
Train logistic regression probes for each layer using balanced data.

Usage:
    python train_probes_balanced.py \
        --activations_path ./outputs/activations_deepseek-math-7b-rl.npy \
        --labels_path ./outputs/labels_deepseek-math-7b-rl.npy \
        --metadata_path ./outputs/metadata_deepseek-math-7b-rl.json \
        --balanced_indices_path ./balanced_outputs/balanced_indices_rl.json \
        --output_dir ./balanced_outputs
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm
from config import (
    PROBE_CONFIG,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DEFAULT_OUTPUT_DIR,
)
from utils import (
    load_json,
    save_json,
    create_data_splits,
    get_model_short_name,
)


def train_probes_balanced(activations_path: str,
                         labels_path: str,
                         metadata_path: str,
                         balanced_indices_path: str,
                         output_dir: str):
    """
    Trains probes using a specific subset of indices for balanced comparison.
    
    Args:
        activations_path: Path to activations .npy file.
        labels_path: Path to labels .npy file.
        metadata_path: Path to metadata JSON.
        balanced_indices_path: JSON file containing indices to use.
        output_dir: Directory for results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading activations from {activations_path}...")
    activations = np.load(activations_path, mmap_mode='r')
    labels = np.load(labels_path)
    metadata = load_json(Path(metadata_path))
    balanced_indices = load_json(Path(balanced_indices_path))
    
    print(f"Original activations shape: {activations.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Balanced indices: {len(balanced_indices)} samples")

    balanced_indices = np.array(balanced_indices)
    
    # Ensure indices are within bounds of activations
    # Activations shape: [n_layers, n_original_samples, hidden_dim]
    max_index = balanced_indices.max() if len(balanced_indices) > 0 else 0
    if max_index >= activations.shape[1]:
        print(f"WARNING: Max balanced index ({max_index}) exceeds activations dimension 1 ({activations.shape[1]}).")
        print("Filtering indices to be within bounds...")
        valid_mask = balanced_indices < activations.shape[1]
        balanced_indices = balanced_indices[valid_mask]
        print(f"Filtered to {len(balanced_indices)} valid indices.")
        
    activations_balanced = activations[:, balanced_indices, :]
    labels_balanced = labels[balanced_indices]
    
    n_layers, n_examples, hidden_dim = activations_balanced.shape
    print(f"Balanced activations shape: {activations_balanced.shape}")
    print(f"Balanced labels shape: {labels_balanced.shape}")
    
    # Create data splits from balanced data
    print(f"\nCreating train/val/test splits ({TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")
    train_idx, val_idx, test_idx = create_data_splits(
        n_examples, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO,
        random_state=PROBE_CONFIG['random_state']
    )
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Calculate class distribution in intersection data
    n_correct = np.sum(labels_balanced)
    n_incorrect = len(labels_balanced) - n_correct
    correct_ratio = n_correct / len(labels_balanced) * 100
    print(f"Class distribution - Correct: {n_correct} ({correct_ratio:.1f}%), Incorrect: {n_incorrect} ({100-correct_ratio:.1f}%)")
    print("Note: Class imbalance handled by class_weight='balanced' in LogisticRegressionCV")
    
    # Train probes for each layer
    results = {
        "model_name": metadata['model_name'],
        "n_layers": n_layers,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_total": n_examples,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "correct_ratio": round(correct_ratio, 1),
        "class_weight": "balanced",
        "indices_path": str(balanced_indices_path),
        "probe_config": PROBE_CONFIG,
        "layer_results": []
    }
    
    print(f"\nTraining probes for {n_layers} layers...")
    print(f"Using {PROBE_CONFIG['cv_folds']}-fold CV for regularization strength")
    
    for layer_idx in tqdm(range(n_layers)):
        # Extract activations for this layer
        X = activations_balanced[layer_idx, :, :]  # Shape: [n_examples, hidden_dim]
        y = labels_balanced
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train probe with cross-validation for regularization
        probe = LogisticRegressionCV(
            Cs=PROBE_CONFIG['C_range'],
            cv=PROBE_CONFIG['cv_folds'],
            max_iter=PROBE_CONFIG['max_iter'],
            random_state=PROBE_CONFIG['random_state'],
            solver=PROBE_CONFIG['solver'],
            class_weight='balanced',
            n_jobs=-1
        )
        
        probe.fit(X_train, y_train)
        
        # Evaluate on all splits
        train_pred = probe.predict(X_train)
        val_pred = probe.predict(X_val)
        test_pred = probe.predict(X_test)
        
        layer_result = {
            "layer": layer_idx,
            "best_C": float(probe.C_[0]),
            "train": {
                "accuracy": float(accuracy_score(y_train, train_pred)),
                "precision": float(precision_score(y_train, train_pred, zero_division=0)),
                "recall": float(recall_score(y_train, train_pred, zero_division=0)),
                "f1": float(f1_score(y_train, train_pred, zero_division=0))
            },
            "val": {
                "accuracy": float(accuracy_score(y_val, val_pred)),
                "precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "recall": float(recall_score(y_val, val_pred, zero_division=0)),
                "f1": float(f1_score(y_val, val_pred, zero_division=0))
            },
            "test": {
                "accuracy": float(accuracy_score(y_test, test_pred)),
                "precision": float(precision_score(y_test, test_pred, zero_division=0)),
                "recall": float(recall_score(y_test, test_pred, zero_division=0)),
                "f1": float(f1_score(y_test, test_pred, zero_division=0))
            }
        }
        
        results["layer_results"].append(layer_result)
    
    # Save results
    model_name = metadata['model_name']
    model_short = get_model_short_name(model_name)
    results_path = output_dir / f"probe_results_balanced_{model_short}.json"
    
    print(f"\nSaving results to {results_path}...")
    save_json(results, results_path)
    
    # Print summary
    test_accuracies = [lr['test']['accuracy'] for lr in results['layer_results']]
    best_layer = np.argmax(test_accuracies)
    best_accuracy = test_accuracies[best_layer]
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Samples: {n_examples} (Correct: {n_correct} [{correct_ratio:.1f}%], Incorrect: {n_incorrect} [{100-correct_ratio:.1f}%])")
    print(f"Class weighting: balanced (sklearn handles imbalance)")
    print(f"Best layer: {best_layer}")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Mean test accuracy: {np.mean(test_accuracies):.4f}")
    print(f"Std test accuracy: {np.std(test_accuracies):.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train logistic probes with balanced data")
    parser.add_argument("--activations_path", type=str, required=True,
                       help="Path to activations .npy file")
    parser.add_argument("--labels_path", type=str, required=True,
                       help="Path to labels .npy file")
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to metadata JSON file")
    parser.add_argument("--balanced_indices_path", type=str, required=True,
                       help="Path to balanced indices JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    
    args = parser.parse_args()
    
    train_probes_balanced(
        args.activations_path,
        args.labels_path,
        args.metadata_path,
        args.balanced_indices_path,
        args.output_dir
    )


if __name__ == "__main__":
    main()
