"""Numbers quoted in the corrected paper's probe text, from the 50/50 balanced summaries.

Per template and model: test accuracy at layer 0, the pre-norm peak and its layer, the
mean over layers 15-29, the first layer above 80% (emergence), and the length baseline.

Usage: python3 probe_paper_numbers.py   (writes out/probe_paper_numbers.csv)
"""
import os

import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.environ.get("RERUN_RESULTS", os.path.join(HERE, "..", "results"))
MODELS = {"deepseek_rl": "DS-RL", "deepseek_instruct": "DS-Instruct", "olmo3_instruct": "Olmo-Instruct",
          "olmo3_thinking_rlvr": "Olmo-Think"}
NEW_RUN = ("deepseek_rl", "olmo3_thinking_rlvr")


def rows():
    base = pd.read_csv(os.path.join(HERE, "out", "trivial_baseline.csv")).set_index("run")
    for t in (270, 584, 785, 884, 922, 942):
        for tag, name in MODELS.items():
            d = "probe_eval_part2" if tag in NEW_RUN else "probe_eval_bal"
            f = os.path.join(RERUN, d, f"{t}_1000_{tag}_tol1_all_bal_summary.csv")
            r = dict(template=t, model=name, n_correct=None, n_incorrect=None, evaluable=False)
            if os.path.exists(f):
                s = pd.read_csv(f)
                pre = s[~s.is_last_layer_postnorm]
                r.update(n_correct=int(s.n_correct[0]), n_incorrect=int(s.n_incorrect[0]), evaluable=bool(s.evaluable.all()))
                if r["evaluable"]:
                    over = pre[pre.acc_mean > 0.8].layer
                    r.update(n_per_class=int(s.n_used[0]) // 2, acc_l0=pre.acc_mean.iloc[0], peak=pre.acc_mean.max(),
                             peak_layer=int(pre.layer[pre.acc_mean.idxmax()]), peak_sd=pre.acc_sd[pre.acc_mean.idxmax()],
                             mean_15_29=pre[pre.layer.between(15, 29)].acc_mean.mean(),
                             emerge=int(over.iloc[0]) if len(over) else None,
                             length_baseline=base.bal_bal_acc.get(f"{t}_1000_{tag}"))
            yield r


if __name__ == "__main__":
    df = pd.DataFrame(rows())
    df.round(3).to_csv(os.path.join(HERE, "out", "probe_paper_numbers.csv"), index=False)
    print(df.round(3).to_string(index=False))
    # Does probe accuracy track model accuracy? The five reported templates, evaluable runs only.
    e = df[df.evaluable & (df.template != 270)]
    print("runs", len(e), spearmanr(e.n_correct / (e.n_correct + e.n_incorrect), e.peak))
