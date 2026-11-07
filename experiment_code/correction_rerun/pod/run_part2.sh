#!/usr/bin/env bash
# Correction rerun, part 2, on one GPU:
#   1. DS-RL probe activations again, now at the "The answer is:" delimiter when the
#      output has no \boxed{ (extract_activations.py, utils.find_phrase_end_position).
#      Its completions are not regenerated: upload them to $RES/<tpl>_deepseek_rl_outputs/.
#   2. Olmo-3-Think probe generation and extraction on the four templates that have
#      figures in the paper, one template at a time, so each finished template is saved
#      before a spend stop. Settings are run_all.sh full: T 0.6, top-p 0.95, 32,768 tokens.
#
# Usage: bash pod/run_part2.sh   (same WORKSPACE / PYTHON / HF_HOME variables as run_all.sh)
# A step with a .done marker in $LOGS is skipped, so a rerun resumes.
set -uo pipefail

WS=${WORKSPACE:-/workspace}
POD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE=$(cd "$POD_DIR/../.." && pwd)/layerwise_linear_probe
PY=${PYTHON:-$WS/venv/bin/python}
export HF_HOME=${HF_HOME:-$WS/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true
RES=$WS/results/part2
LOGS=$WS/logs
STATUS=$LOGS/STATUS
mkdir -p "$RES" "$LOGS"

status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

# step NAME CMD...: runs CMD once; its last output line goes in STATUS.
step() {
	local name=$1 rc last; shift
	if [[ -f $LOGS/$name.done ]]; then status "$name SKIP"; return 0; fi
	status "$name START"
	"$@" >>"$LOGS/$name.log" 2>&1
	rc=$?
	last=$(tail -n 1 "$LOGS/$name.log"); last=${last##*$'\r'}
	if [[ $rc -eq 0 ]]; then touch "$LOGS/$name.done"; status "$name OK ${last:0:300}"
	else status "$name FAILED rc=$rc ${last:0:300}"; fi
}

# extract HF_ID SHORT DIR...: one model load for all DIRs, then a check per DIR.
extract() {
	local id=$1 short=$2 d paths=(); shift 2
	for d in "$@"; do paths+=("$d/completions_$short.json"); done
	(cd "$PROBE" && timeout --kill-after=60 7200 "$PY" extract_activations.py --model_name "$id" \
		--completions_path "${paths[@]}" --output_dir "$@" \
		--probe_position before_answer_with_prompt --batch_size 1 \
		--checkpoint_interval 0 --resume) || return $?
	for d in "$@"; do
		echo -n "$(basename "$d"): "
		"$PY" "$POD_DIR/check_stage.py" extract "$d" "$short" "$d/completions_$short.json" || return 1
	done
}

gen_think() {
	local tpl=$1 out=$RES/${1}_olmo3_thinking_rlvr_outputs
	(cd "$PROBE" && timeout --kill-after=60 14400 "$PY" generate_completions.py \
		--model_name allenai/Olmo-3-7B-Think --dataset_path "./q_data/$tpl.csv" --output_dir "$out" \
		--max_model_len 32768 --max_new_tokens 32768 --temperature 0.6 --top_p 0.95 \
		--tensor_parallel_size 1 --pipeline_parallel_size 1) || return $?
	"$PY" "$POD_DIR/check_stage.py" gen "$out/completions_Olmo-3-7B-Think.json" "$PROBE/q_data/$tpl.csv" ""
}

status "part2 begin"
DS=("$RES"/*_deepseek_rl_outputs)
ds_ok=1
[[ ${#DS[@]} -eq 6 ]] || ds_ok=0
for d in "${DS[@]}"; do
	[[ -f $d/completions_deepseek-math-7b-rl.json ]] || ds_ok=0
	# --resume would skip a dir with old outputs; this step must extract afresh.
	[[ -f $LOGS/part2.extract.deepseek_rl.done ]] || rm -f "$d"/activations_* "$d"/labels_* "$d"/metadata_*
done
if [[ $ds_ok -eq 1 ]]; then
	step part2.extract.deepseek_rl extract deepseek-ai/deepseek-math-7b-rl deepseek-math-7b-rl "${DS[@]}"
else
	status "part2.extract.deepseek_rl FAILED: need 6 DS-RL dirs with completions in $RES"
fi
for tpl in ${THINK_TEMPLATES:-785_1000 584_1000 922_1000 942_1000}; do
	step "part2.gen.think.$tpl" gen_think "$tpl"
	if [[ -f $LOGS/part2.gen.think.$tpl.done ]]; then
		step "part2.extract.think.$tpl" extract allenai/Olmo-3-7B-Think Olmo-3-7B-Think "$RES/${tpl}_olmo3_thinking_rlvr_outputs"
	fi
done
status "part2 end"
