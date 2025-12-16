"""
Extract activations at target positions from generated completions.
Usage:
    python extract_activations.py \
        --model_name deepseek-ai/deepseek-math-7b-rl \
        --completions_path ./outputs/completions_deepseek-math-7b-rl.json \
        --output_dir ./outputs \
        --batch_size 32 \
        --resume
"""

import os
import argparse
import gc
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config import (
    CHECKPOINT_INTERVAL,
    CLEAR_CACHE_BETWEEN_BATCHES,
    DEFAULT_OUTPUT_DIR,
    ENABLE_CHECKPOINTING,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
)
from vllm.transformers_utils.tokenizer import encode_tokens, get_tokenizer
from utils import find_target_position, get_model_short_name, load_json, save_json
from tqdm import tqdm

# Define
Completion = Dict[str, Any]

# Environment variables
hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """Loads tokenizer and model with output_hidden_states=True."""
    print(f"Loading model {model_name}...")
    if model_name in {
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Magistral-Small-2509",
    }:
        tokenizer = get_tokenizer(model_name, tokenizer_mode="mistral")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="cuda",
            cache_dir=hf_cache_dir,
            torch_dtype=torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
            cache_dir=hf_cache_dir,
        )

    model.eval()
    return tokenizer, model


def _resolve_pad_token_id(tokenizer: Any) -> int:
    """Gets pad_token_id, falling back to eos_token_id if necessary."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return eos_token_id

    raise ValueError("Tokenizer must expose pad_token_id or eos_token_id for padding.")


def _filter_valid_examples(completions: List[Completion]) -> List[Completion]:
    """Filters completions to keep only those with valid 'generated_answer'."""
    valid_examples = []
    for comp in completions:
        generated_answer = comp.get("generated_answer")
        if generated_answer and generated_answer != "nan":
            valid_examples.append(comp)
    return valid_examples


def _resume_or_initialize(
    checkpoint_path: Path,
    resume: bool,
    n_layers: int,
    n_valid: int,
    hidden_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Resumes from checkpoint or initializes fresh activation arrays."""
    if resume and checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        start_idx = int(checkpoint["next_idx"])
        print(f"Resuming from index {start_idx}")
        return (
            checkpoint["activations"],
            checkpoint["labels"],
            checkpoint["valid_indices"],
            start_idx,
        )

    activations = np.zeros((n_layers, n_valid, hidden_dim), dtype=np.float32)
    labels = np.zeros(n_valid, dtype=np.int8)
    valid_indices = np.zeros(n_valid, dtype=np.int32)
    return activations, labels, valid_indices, 0


def _save_checkpoint(
    checkpoint_path: Path,
    activations: np.ndarray,
    labels: np.ndarray,
    valid_indices: np.ndarray,
    next_idx: int,
) -> None:
    """Saves intermediate progress to a compressed .npz file."""
    np.savez_compressed(
        checkpoint_path,
        activations=activations,
        labels=labels,
        valid_indices=valid_indices,
        next_idx=next_idx,
    )


def _tokenize_for_probe(
    comp: Completion,
    tokenizer: Any,
    model_name: str,
    probe_position: str,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Tokenizes input and identifies the token index for probing."""

    if probe_position == "question_end":
        question = comp["question"]
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": question}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
        else:
            input_ids = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
            ).input_ids[0]
        target_pos = len(input_ids) - 1
    elif probe_position == "before_answer":
        if model_name in {
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "mistralai/Magistral-Small-2509",
        }:
            input_ids_list = encode_tokens(tokenizer, comp["generated_text"])
            input_ids = torch.tensor(input_ids_list)
        else:
            input_ids = tokenizer(
                comp["generated_text"],
                return_tensors="pt",
                truncation=True,
            ).input_ids[0]

        target_pos = find_target_position(input_ids, tokenizer)
        if target_pos is None:
            target_pos = len(input_ids) - 1
    else:
        raise ValueError(f"Unknown probe_position: {probe_position}")

    if len(input_ids) == 0:
        return None

    return input_ids, target_pos


def _pad_batch(
    batch_input_ids: List[torch.Tensor],
    batch_target_positions: List[int],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Right-pads batch sequences and returns inputs, masks, and aligned positions."""
    max_len = max(len(inp) for inp in batch_input_ids)
    padded_input_ids: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []

    for inp in batch_input_ids:
        padding_length = max_len - len(inp)
        if padding_length > 0:
            pad_tensor = torch.full((padding_length,), pad_token_id, dtype=inp.dtype)
            mask_pad = torch.zeros(padding_length, dtype=torch.long)
            padded = torch.cat([inp, pad_tensor])
            mask = torch.cat([torch.ones(len(inp), dtype=torch.long), mask_pad])
        else:
            padded = inp
            mask = torch.ones(len(inp), dtype=torch.long)

        padded_input_ids.append(padded)
        attention_masks.append(mask)

    return (
        torch.stack(padded_input_ids),
        torch.stack(attention_masks),
        batch_target_positions,
    )


def extract_activations_batched(
    model_name: str,
    completions_path: str,
    output_dir: str,
    probe_position: str = "before_answer",
    batch_size: int = 1,
    resume: bool = False,
):
    """
    Extracts activations from model completions with batching and checkpointing.

    Args:
        model_name: HuggingFace model identifier.
        completions_path: Path to input completions JSON.
        output_dir: Output directory for activations and metadata.
        probe_position: 'question_end' or 'before_answer'.
        batch_size: Batch size for inference.
        resume: Whether to resume from last checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completions = load_json(Path(completions_path))

    model_short = get_model_short_name(model_name)
    activations_path = output_dir / f"activations_{model_short}.npy"
    labels_path = output_dir / f"labels_{model_short}.npy"
    metadata_path = output_dir / f"metadata_{model_short}.json"
    checkpoint_path = output_dir / f"checkpoint_activations_{model_short}.npz"

    if (
        resume
        and activations_path.exists()
        and labels_path.exists()
        and metadata_path.exists()
    ):
        print("\nFound completed activation files")
        print("Extraction already complete. Skipping.")
        return

    # Load model/tokenizer once and gather config metadata for array sizing.
    tokenizer, model = _load_model_and_tokenizer(model_name)
    pad_token_id = _resolve_pad_token_id(tokenizer)

    config = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    print("Filtering valid examples...")
    valid_examples = _filter_valid_examples(completions)

    n_valid = len(valid_examples)
    print(f"Found {n_valid} valid examples")

    if n_valid == 0:
        raise ValueError(
            f"No valid examples found in {completions_path}. "
            "All completions must have a valid generated_answer."
        )

    activations, labels, valid_indices, start_idx = _resume_or_initialize(
        checkpoint_path, resume, n_layers, n_valid, hidden_dim
    )

    print(f"Extracting activations at position: {probe_position}")
    device = "cuda"
    skipped = 0
    pointer = start_idx

    pbar = tqdm(
        total=n_valid, initial=start_idx, desc="Extracting activations", unit="examples"
    )

    while pointer < n_valid:
        batch_end = min(pointer + batch_size, n_valid)
        batch_examples = valid_examples[pointer:batch_end]

        batch_input_ids: List[torch.Tensor] = []
        batch_target_positions: List[int] = []
        batch_valid_indices_local: List[int] = []

        for i, comp in enumerate(batch_examples):
            tokenized = _tokenize_for_probe(comp, tokenizer, model_name, probe_position)
            if tokenized is None:
                skipped += 1
                continue

            input_ids, target_pos = tokenized

            batch_input_ids.append(input_ids)
            batch_target_positions.append(target_pos)
            batch_valid_indices_local.append(pointer + i)

        if len(batch_input_ids) == 0:
            pointer = batch_end
            continue

        input_tensor, attention_mask, adjusted_positions = _pad_batch(
            batch_input_ids, batch_target_positions, pad_token_id
        )
        input_tensor = input_tensor.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states

        # Extract activations for each example in batch
        for local_idx, (global_idx, target_pos) in enumerate(
            zip(batch_valid_indices_local, adjusted_positions)
        ):
            comp = valid_examples[global_idx]

            # Extract from each layer
            layer_activations = []
            # hidden_states[0] is embeddings, hidden_states[1:n_layers+1] are transformer layers
            # hidden_states[-1] may be post-final-norm, reducing probe accuracy
            # We extract exactly n_layers transformer layers (indices 1 through n_layers)
            for layer_idx in range(1, min(n_layers + 1, len(hidden_states))):
                activation = (
                    hidden_states[layer_idx][local_idx, target_pos, :].cpu().numpy()
                )
                layer_activations.append(activation)

            activations[:, global_idx, :] = np.stack(layer_activations)
            valid_indices[global_idx] = comp["index"]
            labels[global_idx] = 1 if comp["is_correct"] else 0

        if CLEAR_CACHE_BETWEEN_BATCHES:
            torch.cuda.empty_cache()
            gc.collect()

        if ENABLE_CHECKPOINTING and (
            batch_end % CHECKPOINT_INTERVAL == 0 or batch_end == n_valid
        ):
            _save_checkpoint(
                checkpoint_path, activations, labels, valid_indices, batch_end
            )

        pointer = batch_end
        pbar.update(len(batch_input_ids))

    pbar.close()

    # Final statistics
    n_examples = n_valid - skipped

    print(f"Probe position: {probe_position}")
    print(f"Valid examples: {n_examples}/{len(completions)}")
    print(f"Activations shape: {activations.shape}")
    print(f"Labels shape: {labels.shape}")
    if len(labels) > 0:
        print(
            f"Positive labels: {labels.sum()}/{len(labels)} ({labels.sum() / len(labels) * 100:.1f}%)"
        )
    else:
        print("Positive labels: 0/0 (0.0%)")

    np.save(activations_path, activations)
    np.save(labels_path, labels)
    metadata = {
        "model_name": model_name,
        "probe_position": probe_position,
        "n_layers": activations.shape[0],
        "n_examples": activations.shape[1],
        "hidden_dim": activations.shape[2],
        "valid_indices": valid_indices.tolist(),
        "n_skipped": skipped,
        "n_correct": int(labels.sum()),
        "accuracy": float(labels.sum() / len(labels)) if len(labels) > 0 else 0.0,
    }

    print(f"Saving metadata to {metadata_path}...")
    save_json(metadata, metadata_path)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Files saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations with batching and checkpointing"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--completions_path", type=str, required=True, help="Path to completions JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--probe_position",
        type=str,
        default="before_answer",
        choices=["question_end", "before_answer"],
        help="Where to extract activations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for extraction (default: 1)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if it exists"
    )

    args = parser.parse_args()

    extract_activations_batched(
        args.model_name,
        args.completions_path,
        args.output_dir,
        args.probe_position,
        args.batch_size,
        args.resume,
    )


if __name__ == "__main__":
    main()
