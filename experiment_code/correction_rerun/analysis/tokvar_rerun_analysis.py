"""Token-variability analysis of the correction rerun (pod_results/full/token_var).

Per problem: mean accuracy over its 50 samples and the output-token CV with the
published bias correction (1 + 1/(4n)). Problems are binned by accuracy exactly as
bargraph_accuracy_vs_token_cv.py does. Three views per model:
  all300   all 300 problems
  pub50    the first 50 problems of the sample, which are the published 50
  notrunc  all 300, samples that hit max_tokens dropped before the CV

The published load_model_data/prepare_bar_data are imported and must reproduce the
all300 numbers from the run_XX.csv files, so the formulas match the paper's.

Usage: python3 tokvar_rerun_analysis.py   (writes out/tokvar_rerun/)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.environ.get("RERUN_RESULTS", os.path.join(HERE, "..", "results"))
RUN = os.path.join(RERUN, "pod_results", "full", "token_var")
# Olmo-3-Think finished its 300 problems in a later run. Its first 2,500 rows are the rows in RUN.
THINK300 = os.path.join(RERUN, "pod_results", "think300", "token_var")
OUT = os.path.join(HERE, "out", "tokvar_rerun")
CODE = os.path.join(HERE, "..", "..", "token_variability_experiment")
PUB = CODE  # the published v1 run CSVs
sys.path.insert(0, CODE)
import bargraph_accuracy_vs_token_cv as bg  # noqa: E402

MODELS = {"deepseek_math_rl_results": "DeepSeek-Math-RL",
          "deepseek_math_instruct_results": "DeepSeek-Math-Instruct",
          "olmo3_instruct_results": "Olmo-3-7B-Instruct",
          "olmo3_thinking_rlvr_results": "Olmo-3-7B-Think"}


def run_dir(d):
    return os.path.join(THINK300 if d == "olmo3_thinking_rlvr_results" else RUN, d)


def per_problem(df):
    g = df.groupby("index").agg(mean_accuracy=("is_correct", "mean"), output_token_mean=("output_tokens", "mean"),
                                output_token_std=("output_tokens", "std"), run_count=("output_tokens", "count"),
                                pos=("pos", "first")).reset_index()
    g["output_token_cv"] = g.output_token_std / g.output_token_mean * (1 + 1 / (4 * g.run_count))
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summary = [], []
    for d, name in MODELS.items():
        jl = os.path.join(run_dir(d), "completions.jsonl")
        if not os.path.exists(jl):
            print(f"{name}: no completions, skipped")
            continue
        # meta.json and the run CSVs are written last. Without them the run was cut
        # short: keep only problems with all their samples and skip the CSV check.
        partial = not os.path.exists(os.path.join(run_dir(d), "meta.json"))
        df = pd.DataFrame([json.loads(line) for line in open(jl)])
        df["is_correct"] = df.is_correct.astype(float)
        if partial:
            n = df.groupby("index").size()
            df = df[df["index"].isin(n[n == n.max()].index)]
            print(f"{name}: partial run, {df['index'].nunique()} complete problems")
        full = per_problem(df)
        if not partial:  # same numbers as the published functions on the run CSVs
            ref = bg.load_model_data(name, run_dir(d)).set_index("index").sort_index()
            mine = full.set_index("index").sort_index()
            assert np.allclose(ref.output_token_cv, mine.output_token_cv) and np.allclose(ref.mean_accuracy, mine.mean_accuracy)
        # The first 50 problems must be the published 50.
        # The published CSVs carry one row with a NaN index, so drop it.
        pub_idx = set(pd.read_csv(glob.glob(os.path.join(PUB, d, "*.csv"))[0])["index"].dropna().astype(int))
        first50 = set(full[full.pos < 50]["index"])
        assert first50 == pub_idx or (partial and first50 < pub_idx), f"{name}: first 50 problems differ from the published set"
        views = {"all300": full, "pub50": full[full.pos < 50].copy(),
                 "notrunc": per_problem(df[df.finish_reason != "length"])}
        for v, g in views.items():
            b = bg.prepare_bar_data({name: g.copy()})
            b["view"] = v
            rows.append(b)
        summary.append(dict(model=name, partial=partial, problems=full["index"].nunique(), samples=len(df), accuracy=df.is_correct.mean(),
                            accuracy_pub50=df[df.pos < 50].is_correct.mean(),
                            truncated=int((df.finish_reason == "length").sum()),
                            mean_out_tokens=df.output_tokens.mean(),
                            mean_cv_all=full.output_token_cv.mean(),
                            distinct_per_problem=df.groupby("index").prediction.nunique().mean()))
    bars = pd.concat(rows, ignore_index=True)
    bars.to_csv(os.path.join(OUT, "bar_data_all_views.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(bars[["view", "model", "accuracy_bin", "count", "avg_cv", "ci_95"]].to_string(index=False))


if __name__ == "__main__":
    main()
