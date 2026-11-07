#!/usr/bin/env bash
# Run the correction rerun on one GPU: token variability, probe generation, and
# probe extraction for each model, in that order.
#
# Usage: bash pod/run_all.sh smoke|full
#
# Each stage writes $WS/logs/<stage>.log, appends one line to $WS/logs/STATUS, and
# on success writes $WS/state/<stage>.done. A rerun skips stages that have a done
# marker. A failed stage is recorded as FAILED and the driver moves on. The script
# exits 1 at the end if any stage failed in this invocation.
#
# Environment (all optional; the defaults are for the RunPod pod):
#   WORKSPACE        root for results, logs, and state (default /workspace)
#   PYTHON           interpreter (default $WORKSPACE/venv/bin/python)
#   HF_HOME          Hugging Face cache (default $WORKSPACE/hf)
#   MODELS           subset of model tags to run, space separated (default: all four)
#   Local smoke tests only, never on the pod:
#   MODEL_OVERRIDE   load this HF id in every stage instead of the paper models
#   SMOKE_MAX_TOKENS, SMOKE_MAX_MODEL_LEN   cap generation length (smoke mode only)
#   GPU_UTIL         --gpu-memory-utilization for tokvar and probe generation
#                    (default: 0.95 for tokvar, the generate_completions.py default for gen)
#   NOTIFY_CMD       shell command run once at the very end, for example to send yourself
#                    a message that the run finished. It never stops the pod.
set -uo pipefail

MODE=${1:-}
if [[ $MODE != smoke && $MODE != full ]]; then
	echo "usage: bash run_all.sh smoke|full" >&2
	exit 2
fi

WS=${WORKSPACE:-/workspace}
POD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS=$(cd "$POD_DIR/../.." && pwd)
CODE=$EXPERIMENTS/token_variability_experiment
PROBE=$EXPERIMENTS/layerwise_linear_probe
PY=${PYTHON:-$WS/venv/bin/python}
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_HUB_OFFLINE=1          # setup_pod.sh downloaded everything; never fetch mid-run
export TOKENIZERS_PARALLELISM=true
LOGS=$WS/logs
STATE=$WS/state
RES=$WS/results/$MODE
mkdir -p "$LOGS" "$STATE" "$RES"
STATUS=$LOGS/STATUS

# tag -> HF id. Tags follow the output-dir names of generate_completion_script.sh.
declare -A HF=(
	[deepseek_rl]=deepseek-ai/deepseek-math-7b-rl
	[deepseek_instruct]=deepseek-ai/deepseek-math-7b-instruct
	[olmo3_instruct]=allenai/Olmo-3-7B-Instruct
	[olmo3_thinking_rlvr]=allenai/Olmo-3-7B-Think
)
# Original probe sampling from generate_completion_script.sh:
# "temperature top_p max_model_len max_new_tokens".
declare -A GEN=(
	[deepseek_rl]="0.7 0.95 4096 4096"
	[deepseek_instruct]="0.7 0.95 4096 4096"
	[olmo3_instruct]="0.6 0.95 32768 32768"
	[olmo3_thinking_rlvr]="0.6 0.95 32768 32768"
)
# DeepSeek first: shorter outputs, so their results arrive early.
ORDER=(${MODELS:-deepseek_rl deepseek_instruct olmo3_instruct olmo3_thinking_rlvr})
TEMPLATES_ALL=(270_1000 584_1000 785_1000 884_1000 922_1000 942_1000)

if [[ $MODE == full ]]; then
	TV_ARGS=(--n-problems 300 --n-samples 50); TV_N=(300 50)
	TEMPLATES=("${TEMPLATES_ALL[@]}"); GEN_LIMIT=()
	# Timeouts guard against a hang; they are not a budget. Olmo-3-Think tokvar is about
	# 34M output tokens: 4 to 5 h on H100 SXM, 7 to 9 h on H100 PCIe.
	T_TOKVAR_DS=7200; T_TOKVAR_OLMO=18000; T_TOKVAR_THINK=43200
	# Olmo-3-Think gen is one generate call that saves nothing until it ends, so a
	# timeout loses the whole stage. 12 h covers about 14M tokens down to ~320 tok/s.
	T_GEN_DS=7200; T_GEN_OLMO=21600; T_GEN_THINK=43200; T_EXTRACT=3600
else
	# 50 samples per problem, as in the full run: the batch is then big enough that the
	# tok/s figure in the tokvar STATUS line says something about full-run throughput.
	TV_ARGS=(--n-problems 3 --n-samples 50); TV_N=(3 50)
	TEMPLATES=(270_1000); GEN_LIMIT=(--limit 5)
	T_TOKVAR_DS=1800; T_TOKVAR_OLMO=1800; T_TOKVAR_THINK=1800; T_GEN_DS=1800; T_GEN_OLMO=1800; T_GEN_THINK=1800; T_EXTRACT=1800
fi
if [[ -n ${MODEL_OVERRIDE:-} && $MODE == full ]]; then
	echo "MODEL_OVERRIDE is for smoke tests only" >&2
	exit 2
fi

FAILED=0

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

# run_stage NAME TIMEOUT_S FUNC [ARGS...]
# FUNC runs the work and the output check. Its last output line is the summary.
run_stage() {
	local name=$1 tmo=$2; shift 2
	if [[ -f $STATE/$name.done ]]; then
		status "$name SKIP (done: $(cat "$STATE/$name.done"))"
		return 0
	fi
	local log=$LOGS/$name.log t0=$SECONDS rc
	echo "===== $(date -u +%FT%TZ) start $name" >>"$log"
	if command -v nvidia-smi >/dev/null; then
		echo "GPU memory used before stage: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >>"$log"
	fi
	status "$name START"
	STAGE_TIMEOUT=$tmo "$@" >>"$log" 2>&1
	rc=$?
	local dt=$((SECONDS - t0)) summary
	summary=$(tail -n 1 "$log")
	summary=${summary##*$'\r'}   # drop tqdm carriage-return progress
	summary=${summary:0:300}
	if [[ $rc -eq 0 ]]; then
		echo "${dt}s $summary" >"$STATE/$name.done"
		status "$name OK ${dt}s $summary"
	else
		FAILED=1
		[[ $rc -eq 124 ]] && summary="timeout after ${tmo}s"
		status "$name FAILED rc=$rc ${dt}s $summary (see $log)"
	fi
	return 0
}

# Run a command under the stage timeout. SIGTERM first, SIGKILL 60 s later.
run() { timeout --kill-after=60 "$STAGE_TIMEOUT" "$@"; }

load_id() { echo "${MODEL_OVERRIDE:-${HF[$1]}}"; }

stage_tokvar() {
	local tag=$1 extra=() dir
	if [[ -n ${MODEL_OVERRIDE:-} ]]; then extra+=(--model-override "$MODEL_OVERRIDE"); fi
	if [[ $MODE == smoke && -n ${SMOKE_MAX_TOKENS:-} ]]; then extra+=(--max-tokens "$SMOKE_MAX_TOKENS"); fi
	if [[ $MODE == smoke && -n ${SMOKE_MAX_MODEL_LEN:-} ]]; then extra+=(--max-model-len "$SMOKE_MAX_MODEL_LEN"); fi
	# A rerun after a timeout or crash resumes from the finished chunks in completions.jsonl.
	(cd "$CODE" && run "$PY" token_var_rerun.py --model "${HF[$tag]}" --out "$RES/token_var" \
		"${TV_ARGS[@]}" --gpu-memory-utilization "${GPU_UTIL:-0.95}" "${extra[@]}") || return $?
	dir=$(cd "$CODE" && "$PY" -c "import token_var_rerun as t; print(t.MODELS['${HF[$tag]}'][0])")
	[[ -n ${MODEL_OVERRIDE:-} ]] && dir+="__override_${MODEL_OVERRIDE##*/}"
	"$PY" "$POD_DIR/check_stage.py" tokvar "$RES/token_var/$dir" "${TV_N[@]}"
}

stage_gen() {
	local tag=$1 tpl paths=() outs=() rc
	read -r temp top_p max_len max_new <<<"${GEN[$tag]}"
	if [[ $MODE == smoke && -n ${SMOKE_MAX_TOKENS:-} ]]; then max_new=$SMOKE_MAX_TOKENS; fi
	if [[ $MODE == smoke && -n ${SMOKE_MAX_MODEL_LEN:-} ]]; then max_len=$SMOKE_MAX_MODEL_LEN; fi
	local extra=()
	if [[ -n ${GPU_UTIL:-} ]]; then extra+=(--gpu-memory-utilization "$GPU_UTIL"); fi
	for tpl in "${TEMPLATES[@]}"; do
		paths+=("./q_data/$tpl.csv")
		outs+=("$RES/probe/${tpl}_${tag}_outputs")
	done
	# One process for all templates of this model: one model load, one generate call.
	(cd "$PROBE" && run "$PY" generate_completions.py --model_name "$(load_id "$tag")" \
		--dataset_path "${paths[@]}" --output_dir "${outs[@]}" "${GEN_LIMIT[@]}" \
		--max_model_len "$max_len" --max_new_tokens "$max_new" \
		--temperature "$temp" --top_p "$top_p" \
		--tensor_parallel_size 1 --pipeline_parallel_size 1 "${extra[@]}") || return $?
	local short; short=$(load_id "$tag"); short=${short##*/}
	rc=0
	for tpl in "${TEMPLATES[@]}"; do
		echo -n "$tpl: "
		"$PY" "$POD_DIR/check_stage.py" gen "$RES/probe/${tpl}_${tag}_outputs/completions_$short.json" \
			"$PROBE/q_data/$tpl.csv" "${GEN_LIMIT[1]:-}" || rc=1
	done
	return $rc
}

# One process per model for all templates: one model load. --resume skips templates
# whose outputs already exist, so a rerun after a failure redoes only the rest.
stage_extract() {
	local tag=$1 tpl short dir paths=() outs=() rc=0
	short=$(load_id "$tag"); short=${short##*/}
	for tpl in "${TEMPLATES[@]}"; do
		dir=$RES/probe/${tpl}_${tag}_outputs
		if [[ ! -f $dir/completions_$short.json ]]; then
			echo "missing $dir/completions_$short.json (generation failed?)"
			return 1
		fi
		paths+=("$dir/completions_$short.json"); outs+=("$dir")
	done
	(cd "$PROBE" && run "$PY" extract_activations.py --model_name "$(load_id "$tag")" \
		--completions_path "${paths[@]}" --output_dir "${outs[@]}" \
		--probe_position before_answer_with_prompt --batch_size 1 \
		--checkpoint_interval 0 --resume) || return $?
	for tpl in "${TEMPLATES[@]}"; do
		dir=$RES/probe/${tpl}_${tag}_outputs
		echo -n "$tpl: "
		"$PY" "$POD_DIR/check_stage.py" extract "$dir" "$short" "$dir/completions_$short.json" || rc=1
	done
	return $rc
}

status "run_all $MODE begin models=${ORDER[*]} override=${MODEL_OVERRIDE:-none}"
for tag in "${ORDER[@]}"; do
	if [[ -z ${HF[$tag]:-} ]]; then
		status "unknown model tag $tag"; FAILED=1; continue
	fi
	if [[ $tag == olmo* ]]; then t_tv=$T_TOKVAR_OLMO; t_gen=$T_GEN_OLMO; else t_tv=$T_TOKVAR_DS; t_gen=$T_GEN_DS; fi
	if [[ $tag == olmo3_thinking_rlvr ]]; then t_tv=$T_TOKVAR_THINK; t_gen=$T_GEN_THINK; fi
	# Olmo-3-Think tokvar is the longest stage (6-9 h). Run it last, after every other
	# stage, so a budget cap or a hang there cannot cost the other results. Its chunks
	# are saved as they finish, and the first chunk is the published 50 problems.
	[[ $tag != olmo3_thinking_rlvr ]] && run_stage "$MODE.tokvar.$tag" "$t_tv" stage_tokvar "$tag"
	run_stage "$MODE.gen.$tag" "$t_gen" stage_gen "$tag"
	run_stage "$MODE.extract.$tag" $((T_EXTRACT * ${#TEMPLATES[@]})) stage_extract "$tag"
done
for tag in "${ORDER[@]}"; do
	[[ $tag == olmo3_thinking_rlvr ]] && run_stage "$MODE.tokvar.$tag" "$T_TOKVAR_THINK" stage_tokvar "$tag"
done

# Manifest and archives. Small results and activations go in separate tars, so the
# small one can be downloaded first.
MANIFEST=$WS/results_manifest.txt
(cd "$WS" && find "results/$MODE" -type f -printf '%s\t%p\n' | sort -k2) >"$MANIFEST"
(cd "$WS" && find "results/$MODE" -type f \( -name '*.csv' -o -name '*.jsonl' -o -name 'meta.json' \
	-o -name 'completions_*.json' -o -name 'metadata_*.json' \) -print0 |
	tar --null -czf "$WS/results_small_$MODE.tar.gz" -T - logs results_manifest.txt) ||
	{ status "tar small FAILED"; FAILED=1; }
# Activations do not compress well, so this tar is uncompressed.
(cd "$WS" && find "results/$MODE" -type f \( -name 'activations_*.npy' -o -name 'labels_*.npy' \
	-o -name 'metadata_*.json' \) -print0 |
	tar --null -cf "$WS/results_activations_$MODE.tar" -T -) ||
	{ status "tar activations FAILED"; FAILED=1; }
ls -l "$WS"/results_*_"$MODE".tar* >>"$STATUS"

if [[ $FAILED -eq 0 ]]; then status "run_all $MODE end: all stages OK"; else status "run_all $MODE end: some stages FAILED"; fi
if [[ -n ${NOTIFY_CMD:-} ]]; then
	bash -c "$NOTIFY_CMD" >>"$LOGS/notify.log" 2>&1 || status "NOTIFY_CMD failed (see $LOGS/notify.log)"
fi
exit $FAILED
