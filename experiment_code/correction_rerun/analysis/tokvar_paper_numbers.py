"""Token-variability numbers quoted in the corrected paper that no other table holds.

Prints the Olmo-3-Think vs Olmo-3-Instruct paired test on their 300 shared problems, their
medians, accuracies and token ratio, and the DS-RL 20-40% bin with truncated samples dropped
and the same problems kept.

Think's first 50 problems ran on an H100 without async scheduling and its other 250 on an
RTX PRO 6000 with it. The comparability table compares those two groups, with Instruct on the
same problems as the control, because Instruct ran all 300 in one setup.

Usage: python3 tokvar_paper_numbers.py   (writes out/tokvar_rerun/paper_numbers.csv, comparability.csv)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from tokvar_rerun_analysis import OUT, run_dir


def load(d):
    return pd.DataFrame([json.loads(line) for line in open(os.path.join(run_dir(d), "completions.jsonl"))])


def cv(df):
    """The published per-problem CV, with its small-sample correction (tokvar_rerun_analysis.per_problem)."""
    g = df.groupby("index").output_tokens
    return g.std() / g.mean() * (1 + 1 / (4 * g.count()))


think_df, instruct_df = load("olmo3_thinking_rlvr_results"), load("olmo3_instruct_results")
think, instruct = cv(think_df), cv(instruct_df)
pair = pd.concat([think, instruct], axis=1).dropna()
rl = load("deepseek_math_rl_results")
acc = rl.groupby("index").is_correct.mean()
bin2040 = acc[(acc > 0.2) & (acc <= 0.4)].index  # the published pd.cut bins
sub = rl[rl["index"].isin(bin2040)]
t, i = pair.iloc[:, 0], pair.iloc[:, 1]
top = think_df.groupby("index").is_correct.mean().gt(0.8) & instruct_df.groupby("index").is_correct.mean().gt(0.8)
top_pair = pair[top.reindex(pair.index)]
row = dict(olmo_shared_problems=len(pair), think_cv=t.mean(), instruct_cv=i.mean(),
           paired_t_p=stats.ttest_rel(t, i).pvalue, paired_t_stat=stats.ttest_rel(t, i).statistic,
           mean_diff_cv=(t - i).mean(), diff_ci95=stats.t.ppf(.975, len(t) - 1) * (t - i).std() / np.sqrt(len(t)),
           wilcoxon_p=stats.wilcoxon(t, i).pvalue, think_higher_problems=int((t > i).sum()),
           top_bin_both_problems=len(top_pair), top_bin_think_cv=top_pair.iloc[:, 0].mean(),
           top_bin_instruct_cv=top_pair.iloc[:, 1].mean(),
           top_bin_paired_t_p=stats.ttest_rel(top_pair.iloc[:, 0], top_pair.iloc[:, 1]).pvalue,
           think_acc=think_df.is_correct.astype(float).mean(), instruct_acc=instruct_df.is_correct.astype(float).mean(),
           think_median=think_df.output_tokens.median(), instruct_median=instruct_df.output_tokens.median(),
           think_mean_tok=think_df.output_tokens.mean(), instruct_mean_tok=instruct_df.output_tokens.mean(),
           min_tok_think=think_df.output_tokens.min(), min_tok_instruct=instruct_df.output_tokens.min(),
           think_truncated=int((think_df.finish_reason == "length").sum()),
           rl_2040_problems=len(bin2040), rl_2040_cv=cv(sub).mean(),
           rl_2040_cv_no_truncated=cv(sub[sub.finish_reason != "length"]).mean())
assert row["olmo_shared_problems"] == 300 and row["rl_2040_problems"] == 9 and abs(row["rl_2040_cv"] - 0.976) < 0.001
pd.DataFrame([row]).to_csv(os.path.join(OUT, "paper_numbers.csv"), index=False, float_format="%.4g")
print(pd.Series(row).map("{:.4g}".format).to_string())

rows = []
for name, df in (("Think", think_df), ("Instruct", instruct_df)):
    df = df.assign(is_correct=df.is_correct.astype(float))
    first = df[df.pos < 50]
    later = df[df.pos >= 50]
    for label, g in (("first50", first), ("later250", later)):
        rows.append(dict(model=name, group=label, problems=g["index"].nunique(), accuracy=g.is_correct.mean(),
                         median_tok=g.output_tokens.median(), mean_tok=g.output_tokens.mean(),
                         mean_cv=cv(g).mean(), median_cv=cv(g).median(),
                         truncated=int((g.finish_reason == "length").sum()),
                         over_8k=int((g.output_tokens > 8000).sum())))
    print(name, "first50 vs later250: CV Welch p =", round(stats.ttest_ind(cv(first), cv(later), equal_var=False).pvalue, 4),
          "| Mann-Whitney p =", round(stats.mannwhitneyu(cv(first), cv(later)).pvalue, 4),
          "| per-problem mean tokens Mann-Whitney p =",
          round(stats.mannwhitneyu(first.groupby("index").output_tokens.mean(), later.groupby("index").output_tokens.mean()).pvalue, 4))
comp = pd.DataFrame(rows)
comp.round(4).to_csv(os.path.join(OUT, "comparability.csv"), index=False)
print(comp.round(3).to_string(index=False))

# The figure's Think vs Instruct gap, with and without the first 50, shows whether the setup change moves the result.
first_idx = think_df[think_df.pos < 50]["index"].unique()
for label, p in (("first50", pair[pair.index.isin(first_idx)]), ("later250", pair[~pair.index.isin(first_idx)])):
    print(label, "paired Think vs Instruct: n =", len(p), "CV", round(p.iloc[:, 0].mean(), 3), round(p.iloc[:, 1].mean(), 3),
          "p =", round(stats.ttest_rel(p.iloc[:, 0], p.iloc[:, 1]).pvalue, 4))


def paired(label, a, b):
    d = a - b
    return dict(check=label, n=len(d), a_cv=a.mean(), b_cv=b.mean(), median_diff=d.median(),
                t_p=stats.ttest_rel(a, b).pvalue, wilcoxon_p=stats.wilcoxon(a, b).pvalue, a_higher=int((d > 0).sum()))


top = top.reindex(pair.index)
first = pair.index.isin(first_idx)
no_trunc = cv(think_df[think_df.finish_reason != "length"]).reindex(pair.index)
short = ~think_df.groupby("index").output_tokens.max().gt(8000).reindex(pair.index)
ds_rl, ds_in = cv(rl), cv(load("deepseek_math_instruct_results"))
ds_top = rl.assign(c=rl.is_correct.astype(float)).groupby("index").c.mean().gt(0.8) \
    & load("deepseek_math_instruct_results").assign(c=lambda x: x.is_correct.astype(float)).groupby("index").c.mean().gt(0.8)
checks = pd.DataFrame([paired("Olmo all", t, i), paired("Olmo top both", t[top], i[top]), paired("Olmo other", t[~top], i[~top]),
                       paired("Olmo top both, first 50", t[top & first], i[top & first]),
                       paired("Olmo top both, later 250", t[top & ~first], i[top & ~first]),
                       paired("Olmo top both, Think truncated dropped", no_trunc[top], i[top]),
                       paired("Olmo top both, no Think sample over 8k", t[top & short], i[top & short]),
                       paired("DeepSeek top both", ds_rl[ds_top], ds_in[ds_top])])
checks.to_csv(os.path.join(OUT, "paired_checks.csv"), index=False, float_format="%.4g")
print(checks.to_string(index=False, float_format="%.4g"))

trend = []
for name, d in (("Olmo-3-Think", "olmo3_thinking_rlvr_results"), ("Olmo-3-Instruct", "olmo3_instruct_results"),
                ("DeepSeek-Math-RL", "deepseek_math_rl_results"), ("DeepSeek-Math-Instruct", "deepseek_math_instruct_results")):
    df = load(d)
    c, a = cv(df), df.assign(x=df.is_correct.astype(float)).groupby("index").x.mean()
    r = stats.spearmanr(c, a)
    trend.append(dict(model=name, spearman=r[0], p=r[1], low_cv=c[a <= 0.8].mean(), low_n=int((a <= 0.8).sum()),
                      high_cv=c[a > 0.8].mean(), high_n=int((a > 0.8).sum())))
trend = pd.DataFrame(trend)
trend.to_csv(os.path.join(OUT, "cv_accuracy_trend.csv"), index=False, float_format="%.4g")
print(trend.to_string(index=False, float_format="%.4g"))
