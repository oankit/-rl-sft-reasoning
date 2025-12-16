"""
Utility functions for probing experiments
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
import math


def check_answer_correct(generated_answer: Optional[str],
                         ground_truth: int,
                         tolerance: int = 1) -> bool:
    """Checks if the generated answer is within tolerance of ground truth."""
    if generated_answer == "nan":
        return False
    
    try:
        return math.isclose(float(generated_answer), ground_truth, abs_tol=tolerance)
    except ValueError:
        return False


def find_target_position(input_ids: torch.Tensor,
                         tokenizer,
                         target_token: str = "{") -> Optional[int]:
    """Finds the index of the target token (e.g., start of \\boxed) in input_ids."""
    # Decode tokens to find \boxed{
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Look for the '{' token that appears after 'boxed' or '\boxed'
    for i in range(len(tokens) - 1, -1, -1):  # Search backwards
        token_str = str(tokens[i])
        if target_token in token_str:
            # Verify it's part of \boxed by checking previous tokens
            if i > 0:
                prev_tokens = ''.join([str(t) for t in tokens[max(0, i-5):i]])
                if 'boxed' in prev_tokens.lower():
                    return i
    
    return None


def save_json(data: dict, path: Path):
    """Saves a dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Loads a dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_data_splits(n_samples: int, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates randomized indices for train, validation, and test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx


def get_model_short_name(model_name: str) -> str:
    """Returns the last component of the model path (e.g. 'deepseek-math-7b-rl')."""
    return model_name.split('/')[-1]