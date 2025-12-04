import csv
import re
import math
from datetime import datetime
from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
from answer_extraction import extract_last_single_answer

# Set Hugging Face cache directory to user's home
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_TOKEN_PATH'] = os.path.expanduser('~/.cache/huggingface/token')


print("Imports loaded successfully!")

# ============ Model Configuration ============
MODEL_NAME = "allenai/Olmo-3-7B-RLZero-Math"
MODEL_CONFIG = {
    "max_tokens": 32768,
    "temperature": 0.6,
}

# ============ Helper Functions ============
def extract_q_ans(answer):
    """Extract the ground truth answer from GSM8K dataset format."""
    num_ans = re.search(r'####\s+(.*)$', answer)
    if num_ans:
        return num_ans.group(1).strip()
    return ''

def extract_answer(question, reasoning):
    """Extract the final answer from model output in string format."""
    output = extract_last_single_answer(question, reasoning)
    match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', output)
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

def format_messages_to_prompt(messages, tokenizer):
    """Format chat messages into a prompt string."""
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        pass
    
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
    
    return "\n\n".join(prompt_parts) + "\n\nAssistant:"

# ============ Evaluation Function ============
def run_evaluation(model_name: str, dataset, llm: LLM, sampling_params: SamplingParams):
    """Evaluate the model on the dataset."""
    print(f"\n")
    print(f"Run - Evaluating: {model_name}")

    csv_filename = f"evaluation_results_{model_name.replace('/', '_')}_GSM8K_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    csv_headers = ['index', 'problem', 'prompt', 'ground_truth', 'prediction', 'extracted', 'is_correct', 'input_tokens', 'output_tokens']
    
    num_correct = 0
    num_prediction = 0
    
    tokenizer = llm.get_tokenizer()
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        
        print(f"\nEvaluating: {model_name} on GSM8K")
        print(f"Total questions: {len(dataset)}")
        
        prompts = []
        indices = []
        ground_truths = []
        problems = []
        
        for idx, row in dataset.iterrows():
            problem = row['question']
            
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": problem}
            ]
            
            prompt = format_messages_to_prompt(messages, tokenizer)
            
            prompts.append(prompt)
            indices.append(row['index'])
            ground_truths.append(row['extracted_answer'])
            problems.append(problem)
            
        # Generate responses
        outputs = llm.generate(prompts, sampling_params)
        
        # Process results
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text
            extracted = extract_answer(problems[i], prediction)
            
            is_correct = False
            try:
                if extracted is not None and extracted.lower() == str(ground_truths[i]).lower():
                    is_correct = True
                    num_correct += 1
            except Exception:
                is_correct = False
            
            num_prediction += 1
            
            csv_row = {
                'index': indices[i],
                'problem': problems[i],
                'prompt': prompts[i],
                'ground_truth': ground_truths[i],
                'prediction': prediction,
                'extracted': extracted,
                'is_correct': is_correct,
                'input_tokens': len(output.prompt_token_ids),
                'output_tokens': len(output.outputs[0].token_ids)
            }
            writer.writerow(csv_row)
            
            if num_prediction % 10 == 0:
                current_accuracy = num_correct / num_prediction
                print(f"Progress: {num_prediction}/{len(dataset)} | Accuracy: {current_accuracy:.2%} ({num_correct}/{num_prediction})")
        
        # Final accuracy
        accuracy = num_correct / num_prediction if num_prediction > 0 else 0
        print(f"\n{'='*60}")
        print(f"Final Accuracy: {accuracy:.2%} ({num_correct}/{num_prediction})")
        print(f"{'='*60}")
        
        accuracy_row = {
            'problem': 'ACCURACY',
            'prompt': '',
            'ground_truth': '',
            'prediction': f"{accuracy:.2%} ({num_correct}/{num_prediction})",
            'is_correct': '',
            'input_tokens': '',
            'output_tokens': ''
        }
        writer.writerow(accuracy_row)
        print(f"Results saved to {csv_filename}")

# ============ Main Execution ============
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# GSM8K Evaluation with vLLM - Olmo-3-7B-RLZero-Math")
    print(f"{'#'*60}\n")
    
    # Load dataset
    complete_dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
    print(f"Loaded {len(complete_dataset)} samples")
    
    pd_dataset = complete_dataset.to_pandas()
    # Sample 50
    pd_dataset = pd_dataset.sample(n=50, random_state=42).reset_index()
    pd_dataset['extracted_answer'] = pd_dataset['answer'].apply(extract_q_ans)
    
    # Initialize vLLM
    sampling_params = SamplingParams(
        temperature=MODEL_CONFIG["temperature"],
        max_tokens=MODEL_CONFIG["max_tokens"]
    )
    
    llm = LLM(model=MODEL_NAME, trust_remote_code=True, gpu_memory_utilization=0.95)
    
    try:
        run_evaluation(MODEL_NAME, pd_dataset, llm, sampling_params)
    except Exception as e:
        print(f"Error during run: {e}")
        import traceback
        traceback.print_exc()
