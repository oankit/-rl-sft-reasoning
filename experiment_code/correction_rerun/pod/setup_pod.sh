#!/usr/bin/env bash
# Set up the RunPod pod for the correction rerun: a pinned Python stack in
# /workspace/venv, the four models and GSM8K-Platinum in /workspace/hf, and a CUDA
# check. Everything lives on the /workspace volume, so it survives a pod restart.
# Safe to rerun; finished downloads are skipped.
#
# Usage: bash pod/setup_pod.sh
set -euo pipefail

WS=${WORKSPACE:-/workspace}
# LOCAL_DIR holds the venv, uv and the model cache. Set it to fast, non-persistent
# container disk (e.g. /root/local) so only logs, state and results go on the
# network volume. Default: same as WS.
L=${LOCAL_DIR:-$WS}
POD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME=$L/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR=$L/uv_cache
export UV_PYTHON_INSTALL_DIR=$L/uv_python
UV=$L/bin/uv
PY=$L/venv/bin/python
mkdir -p "$L/bin" "$HF_HOME" "$WS/logs"

# vLLM 0.11.0 is the first release with Olmo3ForCausalLM (PR #24534). 0.11.2 is the
# last bug-fix release of that line: torch 2.9.0, transformers >= 4.56, < 5.
# transformers 4.57.6 is the last v4 patch. 4.57.3 cannot load a cached tokenizer
# with HF_HUB_OFFLINE=1 (fixed in 4.57.4, PR #42880), and run_all.sh runs offline.
# requirements.lock.txt is the full resolution of these pins, tested on a local GPU.
PINS=(vllm==0.11.2 transformers==4.57.6 datasets==4.4.1 pandas==2.3.3 regex==2025.11.3
	scikit-learn==1.7.2 joblib==1.5.2 hf_transfer==0.1.9 tqdm==4.67.1
	accelerate==1.12.0)   # extract_activations.py loads with device_map="auto"

echo "== GPU and disk"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Newer nvidia-smi prints "CUDA UMD Version" in place of "CUDA Version".
cuda_drv=$(nvidia-smi | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9.]*\).*/\2/p')
echo "driver CUDA version: ${cuda_drv:-unknown}"
# torch 2.9.0+cu128 needs a driver for CUDA 12.8 or newer.
if [[ -z $cuda_drv ]]; then
	echo "WARNING: cannot read the driver CUDA version; the torch check below still runs."
elif [[ $(printf '%s\n' 12.8 "$cuda_drv" | sort -V | head -n1) != 12.8 ]]; then
	echo "ERROR: driver supports CUDA $cuda_drv; need 12.8 or newer. Pick another pod." >&2
	exit 1
fi
# Models 57 GB and venv about 10 GB go on $L; results about 15 GB and tars about 15 GB on $WS.
need_l=${NEED_L_GB:-75}; need_ws=${NEED_WS_GB:-35}; [[ $L == "$WS" ]] && need_l=110 && need_ws=0  # part 2 needs about 5 GB on $WS
for d in "$L:$need_l" "$WS:$need_ws"; do
	free_gb=$(df -BG --output=avail "${d%%:*}" | tail -n1 | tr -dc 0-9)
	echo "free on ${d%%:*}: ${free_gb}G (need ${d##*:}G)"
	if ((free_gb < ${d##*:})); then echo "ERROR: not enough free space on ${d%%:*}" >&2; exit 1; fi
done

echo "== Python stack"
if [[ ! -x $UV ]]; then
	curl -LsSf https://astral.sh/uv/0.9.21/install.sh |
		env UV_INSTALL_DIR="$L/bin" UV_NO_MODIFY_PATH=1 sh
fi
[[ -x $PY ]] || "$UV" venv --python 3.12 "$L/venv"
VIRTUAL_ENV=$L/venv "$UV" pip install -r "$POD_DIR/requirements.lock.txt"
VIRTUAL_ENV=$L/venv "$UV" pip install "${PINS[@]}"   # no-op unless the lock drifted

echo "== Versions and CUDA check"
"$PY" - <<'EOF'
import torch, vllm, transformers, datasets, sklearn, huggingface_hub
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| sklearn", sklearn.__version__,
      "| huggingface_hub", huggingface_hub.__version__)
assert torch.cuda.is_available(), "CUDA not available"
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
print("device", name, "capability", cap)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); assert torch.isfinite(x @ x).all()
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
for a in ("Olmo3ForCausalLM", "LlamaForCausalLM"):
    assert a in archs, f"vLLM lacks {a}"
print("CUDA ok, vLLM has Olmo3ForCausalLM and LlamaForCausalLM")
EOF

echo "== Models (parallel, hf_transfer)"
HFCLI=$L/venv/bin/hf
# deepseek-math-7b-instruct ships only .bin weights; the others ship safetensors.
declare -A INCLUDE=(
	[deepseek-ai/deepseek-math-7b-instruct]="*.json *.bin"
	[deepseek-ai/deepseek-math-7b-rl]="*.json *.safetensors"
	[allenai/Olmo-3-7B-Instruct]="*.json *.safetensors *.jinja *.txt"
	[allenai/Olmo-3-7B-Think]="*.json *.safetensors *.jinja *.txt"
)
pids=()
# ONLY_MODELS (space separated HF ids) limits the download, e.g. to one model.
for m in ${ONLY_MODELS:-"${!INCLUDE[@]}"}; do
	# shellcheck disable=SC2086  # the include list must split into separate globs
	(set -f; "$HFCLI" download "$m" --include ${INCLUDE[$m]} \
		>"$WS/logs/download_${m##*/}.log" 2>&1) &
	pids+=("$!:$m")
done
"$PY" -c "from datasets import load_dataset; print(load_dataset('madrylab/gsm8k-platinum', split='test'))"
dl_fail=0
for p in "${pids[@]}"; do
	if wait "${p%%:*}"; then echo "downloaded ${p#*:}"; else echo "ERROR: download failed for ${p#*:} (see $WS/logs/download_${p##*/}.log)" >&2; dl_fail=1; fi
done
((dl_fail == 0)) || exit 1
du -sh "$HF_HOME/hub"/models--*

echo "== Offline load check (tokenizers and configs from the cache)"
HF_HUB_OFFLINE=1 ONLY_MODELS="${ONLY_MODELS:-}" "$PY" - <<'EOF'
import os
from transformers import AutoConfig, AutoTokenizer
for m in os.environ["ONLY_MODELS"].split() or ["deepseek-ai/deepseek-math-7b-instruct", "deepseek-ai/deepseek-math-7b-rl",
          "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-7B-Think"]:
    c = AutoConfig.from_pretrained(m); t = AutoTokenizer.from_pretrained(m)
    print(m, c.architectures, "chat_template" if t.chat_template else "NO chat_template")
EOF
echo "setup done. Next: bash $POD_DIR/run_all.sh smoke"
