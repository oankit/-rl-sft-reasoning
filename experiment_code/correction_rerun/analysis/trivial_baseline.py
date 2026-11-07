"""Can a probe's score be matched without activations? One classifier per template and model,
same splits and probe as probe_eval.py, on two features: log output length and truncated.

Usage: python3 trivial_baseline.py   (writes out/trivial_baseline.csv)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "layerwise_linear_probe"))
RERUN = os.environ.get("RERUN_RESULTS", os.path.join(HERE, "..", "results"))
from probe_eval import balanced_split, fit_eval, split  # noqa: E402

ROOTS = [os.path.join(RERUN, "pod_results", "part2", "part2"),  # newest first
         os.path.join(RERUN, "pod_results", "part2b", "part2"),
         os.path.join(RERUN, "pod_results", "full", "probe")]


def rows():
    seen = set()
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*_outputs"))):
            name = os.path.basename(d)[:-len("_outputs")]
            metas = glob.glob(os.path.join(d, "metadata_*.json"))
            if name in seen or not metas:
                continue
            seen.add(name)
            short = os.path.basename(metas[0])[len("metadata_"):-len(".json")]
            comps = json.load(open(os.path.join(d, f"completions_{short}.json")))
            valid = json.load(open(metas[0]))["valid_indices"]
            y = np.load(os.path.join(d, f"labels_{short}.npy")).astype(int)
            c = [comps[i] for i in valid]
            assert len(c) == len(y)
            X = np.array([[np.log(len(x["output_token_ids"])), x["finish_reason"] == "length"] for x in c], dtype=float)
            r = dict(run=name, n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), truncated=int(X[:, 1].sum()))
            if min(y.sum(), len(y) - y.sum()) < 10 or min(np.bincount(y[split(y, s)[2]], minlength=2).min() for s in range(5)) < 10:
                yield r  # the floor the paper uses: 10 of each class in every test split
                continue
            for tag, sp in (("unbal", split), ("bal", balanced_split)):
                res = [fit_eval(X, y, *(lambda p: (p[0], p[2]))(sp(y, s)), s, scale=True) for s in range(5)]
                r[f"{tag}_bal_acc"] = round(float(np.nanmean([x["bal_acc"] for x in res])), 3)
                r[f"{tag}_auroc"] = round(float(np.nanmean([x["auroc"] for x in res])), 3)
            yield r


if __name__ == "__main__":
    df = pd.DataFrame(rows()).sort_values("run")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "out", "trivial_baseline.csv"), index=False)
    print(df.to_string(index=False))
