"""
Generate completions for a given model and dataset.
Usage:
    python3 generate_completions.py \
        --model_name mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --dataset_path q_data/942_1000.csv \
        --output_dir ./942_r1_outputs
"""

import argparse
import atexit
import gc
import math
import os
import re
import signal
import sys
import torch
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from vllm import LLM, SamplingParams
from __future__ import annotations
from answer_extraction import extract_last_single_answer
from config import (
    ANSWER_TOLERANCE,
    DEFAULT_OUTPUT_DIR,
    MODEL_CONFIG,
    SYSTEM_PROMPT_TEMPLATE_INSTRUCT,
    SYSTEM_PROMPT_TEMPLATE_MAGISTRAL,
    SYSTEM_PROMPT_TEMPLATE,
)
from utils import (
    check_answer_correct,
    get_model_short_name,
    save_json,
)

# Set HuggingFace cache directory to user's home directory
cache_home = Path.home() / ".cache" / "huggingface"
cache_home.mkdir(parents=True, exist_ok=True)
hub_cache = cache_home / "hub"
hub_cache.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(cache_home.parent)
os.environ["HF_HUB_CACHE"] = str(hub_cache)
os.environ["XDG_CACHE_HOME"] = str(cache_home.parent)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_DOWNLOAD_DIR = Path.home() / ".cache" / "huggingface" / "models"
DEFAULT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)




def _build_system_prompt(model_name: str) -> str:
    """Selects and formats the appropriate system prompt for the model."""
    template = None
    if model_name == "Mistral-Small-3.2-24B-Instruct-2506":
        template = SYSTEM_PROMPT_TEMPLATE_INSTRUCT
    elif model_name == "mistralai/Magistral-Small-2509":
        template = SYSTEM_PROMPT_TEMPLATE_MAGISTRAL
    else:
        template = SYSTEM_PROMPT_TEMPLATE
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return template.format(
        today=today.strftime("%Y-%m-%d"),
        yesterday=yesterday.strftime("%Y-%m-%d")
    )


def create_prompt(math_problem: str) -> str:
    """Constructs the user prompt with instruction to reason step-by-step."""
    return f"{math_problem} Please reason step by step, and put your final answer within \\boxed{{}}."


def extract_answer(question, reasoning):
    """Extracts and parses numerical answer from reasoning text."""
    output = extract_last_single_answer(question, reasoning)
    match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', output)
    if match:
        num_int = int(float(match.group(1)) / float(match.group(2)))
        return str(num_int)
    match = re.search(r'(\d+)\\sqrt\{(\d+)\}', output)
    if match:
        num_int = float(match.group(1)) * math.sqrt(float(match.group(2)))
        return str(int(num_int))
    # 2812+5=2817
    match = re.search(r'(\d+)\+(\d+)=(\d+)', output)
    if match:
        num_int = float(match.group(1)) + float(match.group(2))
        return str(int(num_int))
    else:
        return output


def format_messages_to_prompt(messages, tokenizer):
    """
    Formats chat messages into a single prompt string using tokenizer template or fallback.
    """
    try:
        # Try to use the tokenizer's chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        pass

    # Fallback: manually format messages
    prompt_parts = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'system':
            prompt_parts.append(f"System: {content}")
        elif role == 'user':
            prompt_parts.append(f"User: {content}")
        elif role == 'assistant':
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(content)

    # Add a simple prompt for the assistant to respond
    return "\n\n".join(prompt_parts) + "\n\nAssistant:"


def cleanup():
    """Releases distributed processes and clears GPU memory."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[Cleanup] Warning during process group cleanup: {e}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[Cleanup] Warning during GPU cleanup: {e}")
    
    print("[Cleanup] Cleanup completed.")


def _load_dataset(dataset_path: str, limit: int | None) -> pd.DataFrame:
    """Loads and validates the CSV dataset."""
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    required_columns = {"question", "answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {dataset_path} is missing required columns: {sorted(missing)}"
        )

    df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
    if df["answer"].isnull().any():
        raise ValueError("All answers must be numeric.")

    return df.reset_index(drop=True)


def _merge_generation_overrides(
    model_name: str,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
) -> dict:
    """Merges default model config with command-line overrides."""
    config = MODEL_CONFIG.get(model_name, {}).copy()
    if max_new_tokens is not None:
        config["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p
    return config


def _build_sampling_params(gen_config: dict) -> SamplingParams:
    """Creates vLLM SamplingParams from configuration."""
    sampling_kwargs = {
        "temperature": gen_config.get("temperature", 0.15),
        "max_tokens": gen_config.get("max_new_tokens", 64000),
        "top_p": gen_config.get("top_p", 0.95)
    }
    return SamplingParams(**sampling_kwargs)


def _build_llm_kwargs(
    model_name: str,
    gen_config: dict,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> dict:
    """Constructs initialization arguments for vLLM LLM engine."""
    llm_kwargs = {
        "model": model_name,
        "tokenizer_mode": gen_config.get("tokenizer_mode", "auto"),
        "load_format": gen_config.get("load_format", "auto"),
        "config_format": gen_config.get("config_format", "auto"),
        "limit_mm_per_prompt": gen_config.get("limit_mm_per_prompt", {"image": 0}),
        "trust_remote_code": gen_config.get("trust_remote_code", True),
        "gpu_memory_utilization": gpu_memory_utilization,
        "download_dir": str(DEFAULT_DOWNLOAD_DIR),
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
    }
    if "mistral" in model_name.lower():
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
        llm_kwargs["config_format"] = "mistral"

    return llm_kwargs


def _resolve_index(row, fallback_idx: int) -> int:
    """Gets the question index from row, falling back to enumeration if missing."""
    try:
        if "index" in row.index and not pd.isna(row["index"]):
            return int(row["index"])
    except (KeyError, ValueError, TypeError):
        pass
    return fallback_idx


def generate_completions(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    limit: int | None,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> None:
    """
    Main function to generate completions using vLLM.
    
    Loads dataset, configures LLM, generates responses, extracts answers,
    checks correctness, and saves results to JSON.
    """
    df = _load_dataset(dataset_path, limit)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    merged_config = _merge_generation_overrides(
        model_name, max_new_tokens, temperature, top_p
    )
    sampling_params = _build_sampling_params(merged_config)
    llm_kwargs = _build_llm_kwargs(
        model_name,
        merged_config,
        gpu_memory_utilization,
        max_num_seqs,
        max_model_len,
        tensor_parallel_size,
        pipeline_parallel_size,
    )
    resolved_system_prompt = _build_system_prompt(model_name)

    print(f"Loading {model_name} with vLLM...")
    llm = None
    try:
        llm = LLM(**llm_kwargs)
        tokenizer = llm.llm_engine.tokenizer

        model_short = get_model_short_name(model_name)
        output_path = output_directory / f"completions_{model_short}.json"

        results = []
        n_correct = 0
        n_extracted = 0
        
        print(f"Starting generation for {len(df)} prompts...")

        # Prepare all prompts
        prompts = []
        for _, row in df.iterrows():
            question = str(row["question"])
            user_prompt = create_prompt(question)
            messages = [
                {"role": "system", "content": resolved_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(format_messages_to_prompt(messages, tokenizer))

        # Generate completions for all prompts
        try:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            print(f"Generated {len(outputs)} completions.")
        except Exception as exc:
            print(f"\n[ERROR] Failed during generation: {exc}")
            import traceback
            traceback.print_exc()
            outputs = []

        # Process results
        for i, (_, row) in enumerate(df.iterrows()):
            question = str(row["question"])
            ground_truth = float(row["answer"])
            original_index = row.name

            text = ""
            if i < len(outputs) and outputs[i].outputs:
                text = outputs[i].outputs[0].text
            else:
                print(f"Warning: No output generated for question at index {original_index}")

            extracted_answer = extract_answer(question, text)
            if not extracted_answer:
                extracted_answer = "nan"
            if extracted_answer != "nan":
                n_extracted += 1

            is_correct = check_answer_correct(
                extracted_answer, ground_truth, ANSWER_TOLERANCE
            )
            if is_correct:
                n_correct += 1
            
            record = {
                "index": _resolve_index(row, original_index),
                "question": question,
                "ground_truth": ground_truth,
                "generated_text": text,
                "generated_answer": extracted_answer,
                "is_correct": bool(is_correct),
            }
            results.append(record)

        print(f"\nSaving results to {output_path}...")
        try:
            save_json(results, output_path)
            print("Results saved successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
            import traceback
            traceback.print_exc()
            raise

        total = len(results)
        accuracy = (n_correct / total) * 100 if total else 0.0
        extracted_pct = (n_extracted / total) * 100 if total else 0.0

        print(f"Results saved to: {output_path}")
        print(f"Total questions: {total}")
        print(f"Answers extracted: {n_extracted} ({extracted_pct:.1f}%)")
        print(f"Correct answers: {n_correct} ({accuracy:.1f}%)")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in generate_completions: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup resources
        print("\n[Cleanup] Starting cleanup process...")
        try:
            if llm is not None:
                print("[Cleanup] Deleting LLM object...")
                del llm
                gc.collect()
                print("[Cleanup] LLM object deleted and garbage collected.")
        except Exception as e:
            print(f"[Cleanup] Error deleting LLM: {e}")
        
        try:
            cleanup()
        except Exception as e:
            print(f"[Cleanup] Error during cleanup: {e}")
        
        print("[Cleanup] All cleanup completed.")


def signal_handler(signum, frame):
    """Handles interrupt signals to ensure graceful cleanup."""
    print(f"\n[SIGNAL] Received signal {signum}. Cleaning up...")
    cleanup()
    sys.exit(1)


def main() -> None:
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Generate GSM8K-style completions with vLLM (simplified flow)"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name (must be compatible with vLLM)",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to CSV dataset with 'question' and 'answer' columns",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save completions JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of questions (useful for smoke tests)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16384,
        help="Override max new tokens (falls back to config or 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Override nucleus sampling top_p",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences per iteration",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Model context length",
    )
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95, help='GPU memory utilization')
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="Pipeline parallel size",
    )
    args = parser.parse_args()

    atexit.register(cleanup)

    generate_completions(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )


if __name__ == "__main__":
    main()
