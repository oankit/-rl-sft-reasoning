# How to run the correction rerun on a RunPod H100

This guide runs the rerun behind version 2 of the paper on one H100 80GB pod: the token-variability rerun and probe completions plus activations for four models. Probe training (`layerwise_linear_probe/probe_eval.py`) needs only a CPU, so run it after you stop the pod.

The pod stays billable until you stop it. Nothing in these scripts stops it for you.

## Pick the pod

- GPU: 1x H100 80GB, Secure Cloud.
- Image: any CUDA 12.x PyTorch template. `setup_pod.sh` installs its own pinned stack in `/workspace/venv`, so the image's own PyTorch is not used.
- Driver: CUDA 12.8 or newer. The installed `torch 2.9.0+cu128` needs it, and `setup_pod.sh` exits if the driver is older. Use the CUDA-version filter in the RunPod console.
- Volume: 150 GB at `/workspace`. `setup_pod.sh` exits if less than 110 GB is free.

## Copy the code to the pod

From the repo root, copy the `experiment_code` directory to `/workspace/experiment_code`, for example with the pod's SSH-over-TCP address from the RunPod console:

```bash
COPYFILE_DISABLE=1 tar --no-xattrs --exclude __pycache__ --exclude '*_results' --exclude balanced_outputs --exclude correction_rerun/results -czf experiment_code.tgz experiment_code
scp -P <port> experiment_code.tgz root@<pod-ip>:/workspace/
ssh -p <port> root@<pod-ip> 'cd /workspace && tar xzf experiment_code.tgz'
```

Run the remaining commands on the pod inside `tmux`, so that a dropped SSH connection does not kill the run.

## Set up the pod

```bash
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
tmux new -s rerun
mkdir -p /workspace/logs
bash /workspace/experiment_code/correction_rerun/pod/setup_pod.sh 2>&1 | tee /workspace/logs/setup.log
```

`setup_pod.sh` does the following, and stops at the first error:

1. Checks the driver's CUDA version and the free disk space.
2. Installs uv 0.9.21 and Python 3.12, then installs `requirements.lock.txt` into `/workspace/venv`. The lock pins `vllm==0.11.2`, `transformers==4.57.6`, `torch==2.9.0`, `datasets==4.4.1`, `scikit-learn==1.7.2`, and the rest of the stack.
3. Prints the versions, runs a bf16 matmul on the GPU, and checks that vLLM has `Olmo3ForCausalLM` and `LlamaForCausalLM`.
4. Downloads the four models in parallel with `hf_transfer` into `HF_HOME=/workspace/hf`, about 57 GB in total, and downloads GSM8K-Platinum.
5. Loads each tokenizer and config with `HF_HUB_OFFLINE=1`, to prove the cache is complete.

Expect 10 to 20 minutes, most of it the model download. The last line is `setup done. Next: bash /workspace/experiment_code/correction_rerun/pod/run_all.sh smoke`.

If a gated or rate-limited download fails, run `export HF_TOKEN=<your token>` and rerun `setup_pod.sh`. Finished downloads are skipped.

## Run the smoke test first

```bash
bash /workspace/experiment_code/correction_rerun/pod/run_all.sh smoke
```

The smoke run uses the real models and the real sampling settings on a few prompts:

- Token variability: 3 problems x 50 samples per model. 50 samples is the full-run setting, so the batch is large enough to give a usable throughput figure.
- Probe generation: the first 5 rows of template 270 per model.
- Probe extraction: those 5 completions per model.

Expect 20 to 40 minutes. Most of that time goes to 8 vLLM model loads and the 150 Olmo-3-Think traces. When the run ends, `STATUS` shows `run_all smoke end: all stages OK`. Before you start the full run, read the `OK` lines:

- `tokvar` lines show `acc=`, `truncated=`, `distinct_per_problem=`, `mean_out_tokens=`, and `out_tok_per_s=`. `distinct_per_problem` is the mean number of distinct completions per problem. Expect close to 50. A value well below 50 means the samples repeat, which is the defect this rerun fixes.
- `gen` lines show `correct=`, `exact=`, `no_answer=`, `truncated=`, and `mean_out_tokens=`. `correct` counts answers within 1 of the truth, which is the rule the probe labels use. `exact` counts exact matches.
- `extract` lines show the activation shape and the `\boxed{` rate per class for each template.

Do not extrapolate the full-run time from the smoke stage times. They are mostly model load and the tail of the longest traces. The Olmo-3-Think `out_tok_per_s` is a lower bound on full-run throughput, because only 3 problems run. If it is far below 1,000, check the pod before you start the full run. [Estimate the finish time](#estimate-the-finish-time-during-the-run) from the first chunks of the full run instead.

The smoke run writes to `/workspace/results/smoke` and uses its own done markers, so it never mixes with the full run.

## Start the full run

```bash
bash /workspace/experiment_code/correction_rerun/pod/run_all.sh full
```

The driver works through one model at a time, in this order: `deepseek_rl`, `deepseek_instruct`, `olmo3_instruct`, `olmo3_thinking_rlvr`. For each model it runs three stages:

1. `tokvar`: `token_var_rerun.py` with `--n-problems 300 --n-samples 50`. T=0.6, top-p 0.95, chat template, no custom system prompt. The prompt goes to vLLM as token ids, so the DeepSeek prompts start with one BOS token. Problem i gets seed `1000*i`, and vLLM gives its sample k seed `1000*i + k`. The script generates 50 problems per `generate` call and appends each finished chunk to `completions.jsonl`. If the stage fails, rerunning `run_all.sh` resumes from the finished chunks.
2. `gen`: `layerwise_linear_probe/generate_completions.py` on all 6 templates in one process, one model load, and one `generate` call. The sampling is the original `generate_completion_script.sh` setting. DeepSeek uses T=0.7, top-p 0.95, `max_model_len` 4096, and `max_new_tokens` 4096. Olmo uses T=0.6, top-p 0.95, 32768, and 32768.
3. `extract`: `layerwise_linear_probe/extract_activations.py --probe_position before_answer_with_prompt --batch_size 1 --resume` on all 6 templates in one process, with one model load. A rerun skips the templates whose outputs exist.

"No custom system prompt" still leaves each chat template's own default system text. Olmo-3-7B-Instruct gets "You are a helpful function-calling AI assistant...", and Olmo-3-7B-Think gets "You are OLMo, a helpful function-calling AI assistant built by Ai2..." and an opening `<think>`. DeepSeek has no system text. `meta.json` stores the full `example_prompt` for each model.

Probe generation keeps the original string prompts, as in the published probe runs. vLLM adds a BOS token when it encodes a string, so the DeepSeek probe prompts start with two BOS tokens, as the published ones did. Extraction replays the saved `prompt_token_ids`, so it sees the same tokens as generation. Probe generation now sets the vLLM engine seed to 0. The published runs set no seed.

To run a subset of models, set `MODELS`, for example `MODELS="olmo3_instruct" bash /workspace/experiment_code/correction_rerun/pod/run_all.sh full`.

## Check progress

```bash
tail -f /workspace/logs/STATUS                       # one line per stage start and end
tail -f /workspace/logs/full.gen.olmo3_thinking_rlvr.log   # vLLM progress of one stage
grep 'chunk done' /workspace/logs/full.tokvar.*.log  # tokvar throughput and ETA per chunk
ls /workspace/state/                                 # done markers
nvidia-smi
```

A `STATUS` line looks like this:

```
2026-11-06T02:14:09Z full.gen.deepseek_rl OK 1630s 942_1000: n=1000 correct=612 no_answer=31 truncated=4 mean_out_tokens=402
```

## Handle a failed stage

When a stage fails, `STATUS` gets a `FAILED rc=<code>` line with the log path, and the driver continues with the next stage. Return code 124 means the stage hit its timeout. The extraction stages of a model whose generation failed also fail, because their input file is missing. At the end, `run_all.sh` exits 1 if any stage failed.

To retry, fix the cause and run the same `run_all.sh` command again. Stages with a done marker are skipped. To force a stage to rerun, delete its marker, for example `rm /workspace/state/full.gen.olmo3_instruct.done`.

The timeouts are 2 h for DeepSeek `tokvar` and `gen`, 5 h for Olmo-3-Instruct `tokvar`, 12 h for Olmo-3-Think `tokvar`, 6 h for Olmo-3-Instruct `gen`, 12 h for Olmo-3-Think `gen`, and 6 h for each model's `extract` (1 h per template). The timeouts guard against a hang. They are not a budget. In smoke mode every stage has 30 min.

To get a message when the run ends, set `NOTIFY_CMD` to a shell command, for example `NOTIFY_CMD='curl -s -d "rerun done" ntfy.sh/<your topic>' bash /workspace/experiment_code/correction_rerun/pod/run_all.sh full`. It runs once after the last `STATUS` line. Nothing in the scripts stops the pod.

## Estimated time for the full run

These estimates are rough. The Olmo-3-Think row comes from the published Olmo-3-Think outputs, about 2,300 output tokens per sample, so its `tokvar` stage is about 34M output tokens. Olmo-3-7B uses full multi-head attention, so the KV cache holds only about 60 of these sequences at once, and decoding runs at roughly 1,400 to 2,400 output tokens per second.

| Stage | DeepSeek, each | Olmo-3-Instruct | Olmo-3-Think |
|---|---|---|---|
| `tokvar` (15,000 samples) | 15 to 30 min | 30 to 60 min | 4 to 5 h on H100 SXM, 7 to 9 h on H100 PCIe |
| `gen` (6,000 completions) | 10 to 20 min | 20 to 40 min | 1 to 2 h on SXM, up to 3 h on PCIe |
| `extract` (6 x 1,000 forward passes) | 15 to 25 min | 15 to 30 min | 30 to 60 min |

The total is about 8 to 11 hours on SXM and 12 to 16 hours on PCIe. Olmo-3-Think takes most of it.

### Estimate the finish time during the run

Each `tokvar` chunk prints a line like `chunk done: 100/300 problems, 2150 output tok/s, ETA 85 min`. The ETA assumes the remaining problems are as long as the finished ones. About 30 minutes into `full.tokvar.olmo3_thinking_rlvr`, read the first chunk line. If the ETA is longer than the 12 h timeout, stop the run and check the pod. The Olmo-3-Think `gen` stage is about 14M output tokens, so its time is roughly 14M divided by the same tok/s.

## Read the outputs

```
/workspace/results/full/token_var/<model_dir>/run_00.csv ... run_49.csv, completions.jsonl, meta.json
/workspace/results/full/probe/<template>_<tag>_outputs/completions_<model>.json
/workspace/results/full/probe/<template>_<tag>_outputs/activations_<model>.npy   [layers, N, 4096] float32
/workspace/results/full/probe/<template>_<tag>_outputs/labels_<model>.npy, metadata_<model>.json
/workspace/results_manifest.txt               size and path of every result file
/workspace/results_small_full.tar.gz          CSV, JSONL, completions, metadata, logs, manifest
/workspace/results_activations_full.tar       activations, labels, metadata, about 12 GB
```

`metadata_<model>.json` records `boxed_found` for each example and the `\boxed{` rate per class. The last layer in `activations_<model>.npy` is the state after the final norm, because Hugging Face returns that state as its last hidden state. Drop that layer or flag it in the probe tables.

## Download the results and stop the pod

Download the finished results while the Olmo-3-Think stages run. The DeepSeek and Olmo-3-Instruct stages finish in the first 3 to 4 hours, and their activations are about 9 GB of the 12 GB. Run this on your own machine, after `STATUS` shows `full.extract.olmo3_instruct OK`:

```bash
mkdir -p results
rsync -av -e 'ssh -p <port>' \
  --exclude 'token_var/olmo3_thinking_rlvr_results/' --include 'token_var/***' \
  --include 'probe/' --include 'probe/*deepseek*_outputs/***' --include 'probe/*olmo3_instruct*_outputs/***' \
  --exclude '*' root@<pod-ip>:/workspace/results/full/ results/full/
```

The first rule leaves out the Olmo-3-Think token-variability directory, because the run is still writing it. After the run ends, run `rsync -av -e 'ssh -p <port>' root@<pod-ip>:/workspace/results/full/ results/full/` to fetch the rest, or download the tars. To download the tars, download the small tar first. It holds everything except the activations:

```bash
scp -P <port> root@<pod-ip>:/workspace/results_small_full.tar.gz .
scp -P <port> root@<pod-ip>:/workspace/results_activations_full.tar .
```

Check the downloaded sizes against `results_manifest.txt`. Then stop the pod in the RunPod console. The `/workspace` volume keeps its files while the pod is stopped, but RunPod still bills for the volume until you delete it.

## Run part 2

`run_part2.sh` extracts DeepSeek-Math-RL activations again, at the final "The answer is:" when an output has no `\boxed{`. It reuses the completions from `run_all.sh`, so copy them first. `run_all.sh full` has already generated the Olmo-3-Think probe data, so `THINK_TEMPLATES=" "` skips those steps. The paper's Olmo-3-Think probes came from this script on its four default templates, with no Counting (884) run, so the data from `run_all.sh` adds templates that the paper did not use:

```bash
mkdir -p /workspace/results/part2
cp -r /workspace/results/full/probe/*_deepseek_rl_outputs /workspace/results/part2/
THINK_TEMPLATES=" " bash /workspace/experiment_code/correction_rerun/pod/run_part2.sh
```

The script builds no tar. Download its output before you stop the pod:

```bash
rsync -av -e 'ssh -p <port>' root@<pod-ip>:/workspace/results/part2/ results/part2/
```

The paper's Olmo-3-Think token run used `token_var_rerun.py --async-scheduling` for its last 250 problems. `run_all.sh` does not pass that flag. It changes the engine loop, not the sampling settings.

## Train the probes on a CPU

On a CPU machine, unpack both tars and run `layerwise_linear_probe/probe_eval.py` on each extraction directory:

```bash
mkdir -p probe_eval_bal
python experiment_code/layerwise_linear_probe/probe_eval.py results/full/probe/270_1000_deepseek_instruct_outputs deepseek-math-7b-instruct probe_eval_bal/270_1000_deepseek_instruct_tol1_all_bal --scale --balance
```

## Run a local smoke test without the paper models

The scripts can run against any small model on a local GPU. That checks the plumbing without the pod. This is how the driver was tested on an RTX 5070 Ti:

```bash
WORKSPACE=~/correction_rerun/ws PYTHON=~/correction_rerun/venv/bin/python HF_HOME=~/correction_rerun/hf \
MODEL_OVERRIDE=Qwen/Qwen3-0.6B-Base SMOKE_MAX_TOKENS=512 SMOKE_MAX_MODEL_LEN=2048 GPU_UTIL=0.85 \
bash experiment_code/correction_rerun/pod/run_all.sh smoke
```

`MODEL_OVERRIDE` loads the given model in every stage, and `run_all.sh` refuses it in full mode. The token-variability output directory gets an `__override_<model>` suffix, so an override run cannot pass for a real one.
