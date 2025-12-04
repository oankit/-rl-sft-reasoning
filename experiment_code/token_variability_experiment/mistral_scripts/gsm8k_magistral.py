import csv
import math
import re
import time
import argparse
import atexit
from datetime import datetime, timedelta
from datasets import load_dataset
from vllm import LLM, SamplingParams
from answer_extraction import extract_last_single_answer
import os

print("Imports loaded successfully!")

# Set Hugging Face cache directory to root
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_TOKEN_PATH'] = os.path.expanduser('~/.cache/huggingface/token')

# ============ Model Configuration ============
MODEL_NAME = "mistralai/Magistral-Small-2509"

SYSTEM_PROMPT_TEMPLATE = """First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.
"""

def build_system_prompt() -> str:
    """Build the system prompt for the model."""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today.strftime("%Y-%m-%d"),
        yesterday=yesterday.strftime("%Y-%m-%d"),
    )

# ============ Helper Functions ============
def extract_q_ans(answer):
    """Extract the ground truth answer from GSM8K dataset format."""
    num_ans = re.search(r'####\s+(.*)$', answer)
    if num_ans:
        return num_ans.group(1).strip()
    return ''

def create_prompt(math_problem: str) -> str:
    """Create the prompt for the model."""
    return f"{math_problem} Please reason step by step, and put your final answer within \\boxed{{}}."

def extract_answer(question, reasoning):
    """Extract the final answer from model output in string format."""
    output = extract_last_single_answer(question, reasoning)
    match = re.search(r'\\frac\{(\d+)\}{(\d+)\}', output)
    if match:
        num_int = float(match.group(1)) // float(match.group(2))
        return str(num_int)
    match = re.search(r'(\d+)\\sqrt\{(\d+)\}', output)
    if match:
        num_int = float(match.group(1)) * math.sqrt(float(match.group(2)))
        return str(int(num_int))
    match = re.search(r'(\d+)\+(\d+)=(\d+)', output)
    if match:
        num_int = float(match.group(1)) + float(match.group(2))
        return str(int(num_int))
    else:
        return output

DEFAULT_SAMPLES_PER_RUN = 50

def format_messages_to_prompt(messages, tokenizer):
    """Format chat messages into a prompt string."""
    try:
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


# ============ Evaluation Function ============
def run_evaluation(
    model_name: str,
    dataset,
    llm: LLM,
    sampling_params: SamplingParams,
):
    """
    Run evaluation on the dataset for a single run
    
    Args:
        model_name: Name of the model to evaluate
        dataset: Dataset to evaluate
        llm: Initialized vLLM LLM instance
        sampling_params: Sampling parameters used for generation
        batch_size: Batch size for processing requests
    """
    print(f"\n")
    print(f"Run - Evaluating: {model_name}")

    system_prompt = build_system_prompt()
    total_questions = len(dataset)
    csv_filename = f"evaluation_results_{model_name.replace('/', '_')}_GSM8K_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    csv_headers = ['index', 'problem', 'prompt', 'ground_truth', 'prediction', 'extracted', 'is_correct', 'input_tokens', 'output_tokens']
    
    num_correct = 0
    num_prediction = 0
    prompt_tokens_total = 0
    generation_tokens_total = 0
    
    start_time = time.perf_counter()
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        
        print(f"\nEvaluating: {model_name} on GSM8K")
        print(f"Total questions: {total_questions}")
        
        # Prepare all prompts using tokenizer's chat template
        prompts_list = []
        indices = []
        rows_data = []
        
        # Get tokenizer from LLM engine to format chat messages
        tokenizer = llm.llm_engine.tokenizer
        
        for idx, row in dataset.iterrows():
            problem = row['question']
            user_prompt = create_prompt(problem)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            # Format messages using tokenizer's chat template with fallback
            prompt = format_messages_to_prompt(messages, tokenizer)
            prompts_list.append(prompt)
            indices.append(idx)
            rows_data.append(row)
        
        try:
            outputs = llm.generate(prompts_list, sampling_params=sampling_params)
            
            for i, output in enumerate(outputs):
                idx = indices[i]
                row = rows_data[i]
                problem = row['question']
                ground_truth = row['extracted_answer']
                
                prediction_text = output.outputs[0].text
                extracted = extract_answer(problem, prediction_text)
                
                is_correct = False
                try:
                    if extracted is not None and extracted.lower() == str(ground_truth).lower():
                        is_correct = True
                        num_correct += 1
                except Exception:
                    is_correct = False
                
                num_prediction += 1
                
                # Extract token usage from output metadata
                prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
                output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
                
                prompt_tokens_total += prompt_tokens
                generation_tokens_total += output_tokens
                
                csv_row = {
                    'index': row.get('index', idx),
                    'problem': problem,
                    'prompt': create_prompt(problem),
                    'ground_truth': ground_truth,
                    'prediction': prediction_text,
                    'extracted': extracted,
                    'is_correct': is_correct,
                    'input_tokens': prompt_tokens,
                    'output_tokens': output_tokens
                }
                
                writer.writerow(csv_row)
                csvfile.flush()
                
                current_accuracy = num_correct / num_prediction
                if num_prediction % 10 == 0:
                    print(f"Progress: {num_prediction}/{total_questions} | Accuracy: {current_accuracy:.2%} ({num_correct}/{num_prediction})")
        
        except KeyboardInterrupt:
            print(f"\nEvaluation interrupted at index {idx}")
            print(f"Partial results saved to {csv_filename}")
            if num_prediction > 0:
                print(f"Partial accuracy: {num_correct}/{num_prediction} = {num_correct/num_prediction:.2%}")
            return
        
        # Final accuracy calculation
        if num_prediction > 0:
            accuracy = num_correct / num_prediction
            print(f"\n{'='*60}")
            print(f"Run - Final Accuracy: {accuracy:.2%} ({num_correct}/{num_prediction})")
            print(f"{'='*60}")
            
            # Append accuracy row to CSV
            accuracy_row = {
                'index': '',
                'problem': 'ACCURACY',
                'prompt': '',
                'ground_truth': '',
                'prediction': f"{accuracy:.2%} ({num_correct}/{num_prediction})",
                'extracted': '',
                'is_correct': '',
                'input_tokens': '',
                'output_tokens': ''
            }
            writer.writerow(accuracy_row)
            csvfile.flush()
        
        elapsed_time = time.perf_counter() - start_time
        total_tokens = prompt_tokens_total + generation_tokens_total
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        print(f"Throughput: {tokens_per_sec:.2f} tokens/s ({total_tokens} tokens over {elapsed_time:.1f}s)")
        print(f"Results saved to {csv_filename}")


# ============ Main Execution ============
def main():
    parser = argparse.ArgumentParser(description='Run GSM8K evaluation with vLLM offline inference')
    parser.add_argument('--samples-per-run', type=int, default=DEFAULT_SAMPLES_PER_RUN, help='How many GSM8K samples to evaluate per run')
    parser.add_argument('--random-seed', type=int, default=42, help='Seed for sampling the dataset')
    parser.add_argument('--max-num-batched-tokens', type=int, default=None, help='Maximum number of batched tokens (higher = better GPU utilization but more memory)')
    parser.add_argument('--max-num-seqs', type=int, default=16, help='Maximum number of sequences')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.97, help='GPU memory utilization')
    args = parser.parse_args()

    if args.samples_per_run < 1:
        parser.error("--samples-per-run must be at least 1")

    print(f"\n{'#'*60}")
    print(f"# GSM8K Evaluation with vLLM Offline Inference (Magistral)")
    print(f"# Samples per run: {args.samples_per_run}")
    print(f"{'#'*60}\n")

    # Initialize vLLM engine
    print("Initializing vLLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # max_num_batched_tokens=args.max_num_batched_tokens,  # Dynamic batching: max tokens per batch
        # max_num_seqs=args.max_num_seqs,  # Dynamic batching: max sequences per batch
        tokenizer_mode="mistral",
        load_format="mistral",
        config_format="mistral",
        max_model_len=128000,
    )
    print("vLLM engine initialized successfully!")

    

    # Default sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128000)

    complete_dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
    print(f"Loaded {len(complete_dataset)} samples")

    pandas_dataset = complete_dataset.to_pandas()
    samples_per_run = min(args.samples_per_run, len(pandas_dataset))
    if samples_per_run == 0:
        raise ValueError("No samples available.")
    pd_dataset = pandas_dataset.sample(n=samples_per_run, random_state=args.random_seed).reset_index()
    
    # Isolate the problematic question (index 49) for debugging
    print("DEBUG: Isolating question at index 49 to debug the hang.")
    pd_dataset = pd_dataset.iloc[49:50]
    
    pd_dataset['extracted_answer'] = pd_dataset['answer'].apply(extract_q_ans)

    # Register cleanup function to handle PyTorch distributed process groups
    def cleanup():
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
    
    atexit.register(cleanup)
    
    try:
        run_evaluation(
            MODEL_NAME,
            pd_dataset,
            llm,
            sampling_params,
        )
    except Exception as e:
        print(f"Error during run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup LLM engine and process groups
        try:
            if hasattr(llm, 'llm_engine'):
                llm.llm_engine.shutdown()
        except Exception:
            pass
        cleanup()


if __name__ == "__main__":
    main()
