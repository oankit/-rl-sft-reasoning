"""Check one stage's outputs and print a one-line summary for STATUS.

run_all.sh calls this after each stage. It exits non-zero when the outputs are
missing or incomplete, so a stage that crashed half way never gets a done marker.

  python check_stage.py tokvar  <model_out_dir> <n_problems> <n_samples>
  python check_stage.py gen     <completions_json> <dataset_csv> [limit]
  python check_stage.py extract <extract_dir> <model_short> <completions_json>
"""
import csv
import glob
import json
from collections import Counter
import os
import sys

import numpy as np


def fail(msg):
    print(f"CHECK FAILED: {msg}")
    sys.exit(1)


def tokvar(out_dir, n_problems, n_samples):
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        fail(f"missing {meta_path}")
    meta = json.load(open(meta_path))
    with open(os.path.join(out_dir, "completions.jsonl")) as fh:
        n_lines = sum(1 for _ in fh)
    runs = glob.glob(os.path.join(out_dir, "run_*.csv"))
    if n_lines != n_problems * n_samples:
        fail(f"completions.jsonl has {n_lines} lines, expected {n_problems * n_samples}")
    if len(runs) != n_samples:
        fail(f"{len(runs)} run_*.csv files, expected {n_samples}")
    tps = meta.get("output_tok_per_s")
    print(f"acc={meta['accuracy']:.3f} truncated={meta['truncated']}/{n_lines} "
          f"distinct_per_problem={meta['mean_distinct_samples_per_problem']:.1f} "
          f"mean_out_tokens={meta['mean_output_tokens']:.0f} "
          f"out_tok_per_s={'na' if tps is None else f'{tps:.0f}'}")


def gen(completions_json, dataset_csv, limit=None):
    if not os.path.exists(completions_json):
        fail(f"missing {completions_json}")
    recs = json.load(open(completions_json))
    with open(dataset_csv, newline="") as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    expected = min(n_rows, int(limit)) if limit else n_rows
    if len(recs) != expected:
        fail(f"{len(recs)} records, expected {expected}")
    empty = sum(1 for r in recs if not r.get("output_token_ids"))
    if empty:
        fail(f"{empty} records have no output tokens")
    n = len(recs)
    correct = sum(r["is_correct"] for r in recs)
    # The probe labels accept answers within 1 of the truth; exact= counts exact matches.
    def exact(r):
        try:
            return float(r["generated_answer"]) == float(r["ground_truth"])
        except ValueError:
            return False
    n_exact = sum(exact(r) for r in recs)
    no_answer = sum(r["generated_answer"] == "nan" for r in recs)
    truncated = sum(r["finish_reason"] == "length" for r in recs)
    mean_out = sum(len(r["output_token_ids"]) for r in recs) / n
    print(f"n={n} correct={correct} exact={n_exact} no_answer={no_answer} truncated={truncated} "
          f"mean_out_tokens={mean_out:.0f}")


def extract(extract_dir, model_short, completions_json):
    paths = {k: os.path.join(extract_dir, f"{k}_{model_short}.{ext}")
             for k, ext in [("activations", "npy"), ("labels", "npy"), ("metadata", "json")]}
    for p in paths.values():
        if not os.path.exists(p):
            fail(f"missing {p}")
    acts = np.load(paths["activations"], mmap_mode="r")
    labels = np.load(paths["labels"])
    meta = json.load(open(paths["metadata"]))
    n_valid = sum(1 for r in json.load(open(completions_json))
                  if r.get("generated_answer") and r["generated_answer"] != "nan")
    if acts.shape[1] != n_valid or len(labels) != n_valid:
        fail(f"activations {acts.shape}, labels {labels.shape}, expected {n_valid} examples")
    if meta.get("n_skipped"):
        fail(f"{meta['n_skipped']} examples skipped; their activation rows are zeros")
    if not np.isfinite(acts[:, : min(50, n_valid)]).all():
        fail("non-finite activations")
    anchors = meta.get("probe_anchor")
    if anchors is None or len(anchors) != n_valid:  # stale metadata from before the anchor fix
        fail(f"probe_anchor missing or wrong length in {paths['metadata']}")
    def rate(x):
        return "na" if x is None else f"{x:.3f}"
    print(f"shape={tuple(acts.shape)} pos={int(labels.sum())} neg={int(len(labels) - labels.sum())} "
          f"boxed_rate_correct={rate(meta['boxed_found_rate_correct'])} "
          f"boxed_rate_incorrect={rate(meta['boxed_found_rate_incorrect'])} "
          f"anchors={dict(sorted(Counter(anchors).items()))}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "tokvar":
        tokvar(args[0], int(args[1]), int(args[2]))
    elif cmd == "gen":
        gen(*args)
    elif cmd == "extract":
        extract(*args)
    else:
        fail(f"unknown command {cmd}")
