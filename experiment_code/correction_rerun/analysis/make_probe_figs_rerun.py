"""Probe figures (Figure 1 and the appendix per-template figures) from the correction rerun.

Draws each template with the published visualize_results.plot_layer_performance_regression,
fed the rerun per-layer means over 5 split seeds. Two variants, only runs with at least
10 test examples per class:
  balanced  raw test accuracy on the 50/50 subsample (--balance), the paper's stated method
  balacc    balanced accuracy on all examples, the unbalanced check

DS-RL and Olmo-3-Think come from probe_eval_part2 (DS-RL at the "The answer is" position),
the other two from probe_eval_bal and probe_eval_results. A missing run is skipped.
The function reads only model_name and
layer_results[].layer / .test.<metric>, so those are all this script builds.

Usage: python3 make_probe_figs_rerun.py   (writes ../figures_rerun/probes/<variant>/<template>/)
"""
import inspect
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.environ.get("RERUN_RESULTS", os.path.join(HERE, "..", "results"))
OUT = Path(HERE, "..", "figures_rerun", "probes")
# utils.py imports torch only for type hints; a stub avoids installing it.
_t = types.ModuleType("torch")
_t.Tensor = type("Tensor", (), {})  # matplotlib isinstance-checks torch.Tensor, so not object
sys.modules.setdefault("torch", _t)
sys.path.insert(0, os.path.join(HERE, "..", "..", "layerwise_linear_probe"))
import visualize_results as vr  # noqa: E402

MODELS = {"deepseek_rl": "deepseek-ai/deepseek-math-7b-rl",
          "deepseek_instruct": "deepseek-ai/deepseek-math-7b-instruct",
          "olmo3_instruct": "allenai/Olmo-3-7B-Instruct",
          "olmo3_thinking_rlvr": "allenai/Olmo-3-7B-Think"}
NEW_RUN = ("deepseek_rl", "olmo3_thinking_rlvr")  # in probe_eval_part2, both variants
TEMPLATES = (270, 584, 785, 884, 922, 942)

# The published function hard-codes metric = "accuracy" and puts its legend over the low points.
# Run the same source with the legend below the axes and, for balacc, the metric swapped.
_src = inspect.getsource(vr.plot_layer_performance_regression)
_legend = 'ax.legend(fontsize=18, loc="lower right", framealpha=0.95)'
assert 'metric = "accuracy"' in _src and 'metric_name = "Accuracy"' in _src and _legend in _src
_src = _src.replace(_legend, 'ax.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)')
_acc, _bal = dict(vars(vr)), dict(vars(vr))
exec(_src, _acc)
exec(_src.replace('metric = "accuracy"', 'metric = "balanced_accuracy"')
     .replace('metric_name = "Accuracy"', 'metric_name = "Balanced Accuracy"'), _bal)
PLOT = {"balanced": _acc["plot_layer_performance_regression"], "balacc": _bal["plot_layer_performance_regression"]}


def results(template, tag, variant):
    d = "probe_eval_part2" if tag in NEW_RUN else "probe_eval_bal" if variant == "balanced" else "probe_eval_results"
    f = os.path.join(RERUN, d, f"{template}_1000_{tag}_tol1_all{'_bal' if variant == 'balanced' else ''}_summary.csv")
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f)
    return dict(model_name=MODELS[tag], evaluable=bool(s.evaluable.all()),
                layer_results=[dict(layer=int(r.layer), test=dict(accuracy=r.acc_mean, balanced_accuracy=r.bal_acc_mean))
                               for r in s.itertuples()])


def main():
    for t in TEMPLATES:
        for variant, plot in PLOT.items():
            keep = [r for r in (results(t, tag, variant) for tag in MODELS) if r and r["evaluable"]]
            out = OUT / variant / str(t)
            out.mkdir(parents=True, exist_ok=True)
            plot(keep, out)
            print(variant, t, [r["model_name"].split("/")[-1] for r in keep])


if __name__ == "__main__":
    main()
