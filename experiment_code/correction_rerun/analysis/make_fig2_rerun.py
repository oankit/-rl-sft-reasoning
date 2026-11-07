"""Figure 2 from the correction rerun, drawn by the published bargraph_accuracy_vs_token_cv.main().

For each view, writes one run_XX.csv per sample k from completions.jsonl (the layout
token_var_rerun.py writes, and the only layout main() reads), then runs main() unchanged.

  all300  all 300 problems
  pub50   the first 50 problems, which are the published 50

Usage: python3 make_fig2_rerun.py   (writes ../figures_rerun/<view>/figures/)
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from tokvar_rerun_analysis import MODELS, bg, run_dir  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures_rerun")


def stage(view):
    root = os.path.join(OUT, view)
    for d in MODELS:
        df = pd.DataFrame([json.loads(line) for line in open(os.path.join(run_dir(d), "completions.jsonl"))])
        n = df.groupby("index").size()
        df = df[df["index"].isin(n[n == n.max()].index)]  # complete problems only (partial run)
        if view == "pub50":
            df = df[df.pos < 50]
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)  # main() globs *.csv, so no stale runs
        os.makedirs(os.path.join(root, d))
        for k, g in df.sort_values("pos").groupby("sample"):
            g[["index", "is_correct", "output_tokens"]].to_csv(os.path.join(root, d, f"run_{k:02d}.csv"), index=False)
    return root


def main():
    for view in ("all300", "pub50"):
        root = stage(view)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            bg.main()
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
