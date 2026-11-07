"""Token-variability rerun: one harness for all four models.

Fixes relative to the published runs:
- One prompt format for every model: the DeepSeekMath model-card instruction in
  the user turn, the model's own chat template, no custom system prompt.
- The same sampling for every model (T, top-p), with one seeded request per problem
  that draws n distinct samples (SamplingParams(n=..., seed=...)). The old runs
  started one process per run with no seed, so whole runs repeated. Each problem
  gets its own seed, so sample k of different problems does not share a stream.
- The prompt goes to vLLM as token ids from the chat template, so DeepSeek gets one
  BOS token, not two.
- Problems run in chunks, and each finished chunk is appended to completions.jsonl.
  A rerun with the same settings skips the finished problems.
- Numeric grading (thousands separators, $, %, \\text{}, trailing unit words).
- Raw completions are saved.

Output layout matches bargraph_accuracy_vs_token_cv.py: <out>/<model_dir>/run_XX.csv
(one CSV per sample index, 'index', 'is_correct', 'output_tokens' columns), plus
<out>/<model_dir>/completions.jsonl with every sample.

Usage:
  python token_var_rerun.py --model allenai/Olmo-3-7B-Think --out results --n-problems 300
  python token_var_rerun.py --selfcheck
  # Smoke test with a small model; the output dir gets a suffix so it cannot pass for a real run:
  python token_var_rerun.py --model deepseek-ai/deepseek-math-7b-rl --model-override Qwen/Qwen3-0.6B-Base \
      --max-tokens 256 --max-model-len 2048 --n-problems 3 --n-samples 4
"""
import argparse
import csv
import json
import os
import re
import time

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
MODELS = {  # HF id -> (output dir, max new tokens, max model len)
    "deepseek-ai/deepseek-math-7b-instruct": ("deepseek_math_instruct_results", 3800, 4096),
    "deepseek-ai/deepseek-math-7b-rl": ("deepseek_math_rl_results", 3800, 4096),
    "allenai/Olmo-3-7B-Instruct": ("olmo3_instruct_results", 32768, 36864),
    "allenai/Olmo-3-7B-Think": ("olmo3_thinking_rlvr_results", 32768, 36864),
}
SEED_STRIDE = 1000  # problem i uses seeds seed + 1000*i + k for its samples k < 1000


def gsm8k_truth(answer_field):
    m = re.search(r"####\s+(.*)$", answer_field)
    return m.group(1).strip() if m else ""


def to_number(s):
    """Parse a GSM8K-style answer. Returns None when it is not a single number."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,!;: ]", "", s)            # LaTeX spacing commands
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\", "").strip()
    # A number, optionally followed by one unit word ("91 trees", "16years").
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[A-Za-z]*\.?", s)
    return float(m.group(1)) if m else None


def is_correct(extracted, truth):
    a, b = to_number(extracted), to_number(truth)
    return a is not None and b is not None and abs(a - b) < 1e-6


def build_prompt(tokenizer, question):
    """Return (prompt text, prompt token ids). vLLM gets the token ids.

    vLLM encodes a string prompt with add_special_tokens=True. The DeepSeek chat
    template already starts with BOS, so a string prompt would get two BOS tokens.
    Tokenizing through the template gives exactly the template's tokens.
    """
    msgs = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = list(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    else:
        # DeepSeekMath model-card format, used only if the tokenizer has no template.
        text = f"User: {question}\n{INSTRUCTION}\n\nAssistant:"
        ids = tokenizer.encode(text)
    bos = getattr(tokenizer, "bos_token_id", None)
    assert bos is None or ids[:2] != [bos, bos], "prompt starts with two BOS tokens"
    return text, ids


def load_problems(n):
    from datasets import load_dataset
    df = load_dataset("madrylab/gsm8k-platinum", split="test").to_pandas()
    # Same call as the published runs; the first 50 rows of n=300 are the published 50.
    df = df.sample(n=n, random_state=42).reset_index()
    df["truth"] = df["answer"].apply(gsm8k_truth)
    return df


CSV_FIELDS = ["index", "sample", "ground_truth", "extracted", "is_correct", "output_tokens",
              "input_tokens", "finish_reason"]


def load_finished(path, n_samples):
    """Read completions.jsonl from an interrupted run. Keep only complete problems."""
    by_pos = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn last line from a killed run
                by_pos.setdefault(r["pos"], []).append(r)
    return {p: rs for p, rs in by_pos.items() if len(rs) == n_samples}


def main(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from answer_extraction import extract_last_single_answer

    out_dir, max_new, max_len = MODELS[a.model]
    # Smoke-test overrides. Real runs pass none of these and keep the MODELS settings.
    load_model = a.model_override or a.model
    if a.model_override:
        out_dir += "__override_" + a.model_override.split("/")[-1]
    max_new = a.max_tokens or max_new
    max_len = a.max_model_len or max_len
    out_dir = os.path.join(a.out, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    assert a.n_samples <= SEED_STRIDE

    # Resume: a rerun with the same settings keeps the finished chunks. Problem i
    # (position in the sample) always gets seed + SEED_STRIDE*i, so the chunking and
    # the restart point do not change which random stream a problem draws from.
    cfg = dict(model=a.model, loaded_model=load_model, n_problems=a.n_problems,
               n_samples=a.n_samples, temperature=a.temperature, top_p=a.top_p,
               max_tokens=max_new, max_model_len=max_len, seed=a.seed,
               seed_scheme=f"problem i gets seed + {SEED_STRIDE}*i; vLLM gives its sample k that + k")
    cfg_path = os.path.join(out_dir, "run_config.json")
    jsonl = os.path.join(out_dir, "completions.jsonl")
    if os.path.exists(cfg_path) or os.path.exists(jsonl):
        old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
        if old != cfg:
            raise SystemExit(f"{out_dir} holds a run with other settings ({old}). Delete it to start over.")
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1)
    finished = load_finished(jsonl, a.n_samples)
    with open(jsonl + ".tmp", "w") as fh:  # drop any partial problem, then append
        for p in sorted(finished):
            for r in finished[p]:
                fh.write(json.dumps(r) + "\n")
    os.replace(jsonl + ".tmp", jsonl)

    df = load_problems(a.n_problems)
    todo = [i for i in range(len(df)) if i not in finished]
    print(f"{len(finished)} problems already done, {len(todo)} to generate")
    prompts = None
    if todo:
        llm = LLM(model=load_model, trust_remote_code=True, max_model_len=max_len,
                  gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed,
                  async_scheduling=a.async_scheduling)
        tok = llm.get_tokenizer()
        prompts = [build_prompt(tok, q) for q in df["question"]]
        t_gen, n_tok = 0.0, 0
        for c in range(0, len(todo), a.chunk_problems):
            chunk = todo[c:c + a.chunk_problems]
            sps = [SamplingParams(n=a.n_samples, temperature=a.temperature, top_p=a.top_p,
                                  max_tokens=max_new, seed=a.seed + SEED_STRIDE * i) for i in chunk]
            t0 = time.time()
            outputs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i][1]) for i in chunk], sps)
            t_gen += time.time() - t0
            with open(jsonl, "a") as fh:
                for i, out in zip(chunk, outputs):
                    row = df.iloc[i]
                    for k, s in enumerate(out.outputs):
                        ext = extract_last_single_answer(row["question"], s.text)
                        n_tok += len(s.token_ids)
                        rec = dict(index=int(row["index"]), sample=k, ground_truth=row["truth"],
                                   extracted=ext, is_correct=is_correct(ext, row["truth"]),
                                   output_tokens=len(s.token_ids),
                                   input_tokens=len(out.prompt_token_ids),
                                   finish_reason=s.finish_reason)
                        fh.write(json.dumps(dict(rec, pos=i, prompt=prompts[i][0],
                                                 prediction=s.text)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            done = c + len(chunk)
            rate = n_tok / t_gen
            print(f"chunk done: {done}/{len(todo)} problems, {rate:.0f} output tok/s, "
                  f"ETA {t_gen / done * (len(todo) - done) / 60:.0f} min", flush=True)

    recs = [r for rs in load_finished(jsonl, a.n_samples).values() for r in rs]
    recs.sort(key=lambda r: (r["pos"], r["sample"]))
    assert len(recs) == a.n_problems * a.n_samples, len(recs)
    for k in range(a.n_samples):
        with open(os.path.join(out_dir, f"run_{k:02d}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(r for r in recs if r["sample"] == k)

    by_pos = {}
    for r in recs:
        by_pos.setdefault(r["pos"], set()).add(r["prediction"])
    # async_scheduling stays out of cfg: it changes the engine loop, not the settings a resume must match.
    meta = dict(cfg, async_scheduling=a.async_scheduling,
                accuracy=sum(r["is_correct"] for r in recs) / len(recs),
                truncated=sum(r["finish_reason"] == "length" for r in recs),
                mean_distinct_samples_per_problem=sum(map(len, by_pos.values())) / len(by_pos),
                mean_output_tokens=sum(r["output_tokens"] for r in recs) / len(recs),
                # Throughput of this invocation only (None if it generated nothing).
                output_tok_per_s=(n_tok / t_gen) if prompts and t_gen else None,
                example_prompt=recs[0]["prompt"])
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "example_prompt"}, indent=1))


def selfcheck():
    assert gsm8k_truth("blah\n#### 43,500") == "43,500"
    for ext, truth in [("43500", "43,500"), ("$18", "18"), ("24\\%", "24"), ("\\text{1}", "1"),
                       ("91trees", "91"), ("16 years", "16"), ("1,875.0", "1875")]:
        assert is_correct(ext, truth), (ext, truth)
    for ext, truth in [("26", "24"), ("15wonand7lost", "15"), ("", "3"), ("3/4", "1")]:
        assert not is_correct(ext, truth), (ext, truth)

    class Tok:  # a template that starts with BOS, like DeepSeek's
        chat_template, bos_token_id = "x", 1
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return [1, 7, 8] if tokenize else "<bos>User: ..."
    assert build_prompt(Tok(), "q")[1] == [1, 7, 8]
    Tok.apply_chat_template = lambda self, m, tokenize, add_generation_prompt: [1, 1, 7] if tokenize else ""
    try:
        build_prompt(Tok(), "q")
        raise RuntimeError("double BOS not caught")
    except AssertionError:
        pass

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.jsonl")
        with open(path, "w") as fh:
            for pos, k in [(0, 0), (0, 1), (1, 0)]:
                fh.write(json.dumps(dict(pos=pos, sample=k)) + "\n")
            fh.write('{"pos": 1, "sam')
        assert list(load_finished(path, 2)) == [0]
    print("selfcheck ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--out", default="token_var_results")
    p.add_argument("--n-problems", type=int, default=300)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-problems", type=int, default=50,
                   help="problems per generate call; finished chunks survive a crash or timeout")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--async-scheduling", action="store_true",
                   help="vLLM async_scheduling: schedule the next step while the GPU runs this one")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--model-override", help="smoke tests only: load this HF id instead of --model")
    p.add_argument("--max-tokens", type=int, help="smoke tests only: override max new tokens")
    p.add_argument("--max-model-len", type=int, help="smoke tests only: override context length")
    a = p.parse_args()
    selfcheck() if a.selfcheck or not a.model else main(a)
