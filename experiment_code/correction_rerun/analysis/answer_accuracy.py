"""How many probe-completion answers are actually right, per template and model.

Each answer is compared with the exact answer to the question text (layerwise_linear_probe/label_truth.py):
  right      fixed rule: an integer within 1 of the exact value, or the generator's
             floor-each-step label where that is a valid reading
  exact      equals the exact value (only possible where the exact value is an integer)
  off_le10   wrong, but within 10 of the exact value
  off_gt10   wrong by more than 10
  none       the answer does not parse as a number, after stripping commas, $ and %
plus tol1, the published label: float() on the raw answer, so "51,552" and "36%" count as wrong.
A second table splits each template into answer-size quartiles.

Usage: python3 answer_accuracy.py   (writes out/answer_accuracy.csv, out/answer_accuracy_by_size.csv)
"""
import json
import math
import os
import sys
from fractions import Fraction as F

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "..", "layerwise_linear_probe")
RERUN = os.environ.get("RERUN_RESULTS", os.path.join(HERE, "..", "results"))
POD = os.path.join(RERUN, "pod_results")
RUNS = [os.path.join(POD, "full", "probe"), os.path.join(POD, "part2", "part2"), os.path.join(POD, "part2b", "part2")]
sys.path.insert(0, PROBE)
from label_truth import exact_value, is_correct_fixed, label_is_reading  # noqa: E402

MODELS = {"deepseek_rl": "deepseek-math-7b-rl", "deepseek_instruct": "deepseek-math-7b-instruct",
          "olmo3_instruct": "Olmo-3-7B-Instruct", "olmo3_thinking_rlvr": "Olmo-3-7B-Think"}
TEMPLATES = (270, 584, 785, 884, 922, 942)


def num(s):
    try:
        v = float(str(s).replace(",", "").replace("$", "").rstrip("%"))
    except ValueError:
        return None
    return v if math.isfinite(v) else None  # "nan" and "inf" count as no answer


def published_tol1(s, label):
    """utils.check_answer_correct, without its torch import."""
    try:
        return math.isclose(float(s), label, abs_tol=1)
    except (TypeError, ValueError):
        return False


def rows():
    q = {t: pd.read_csv(os.path.join(PROBE, "q_data", f"{t}_1000.csv")) for t in TEMPLATES}
    for t in TEMPLATES:
        for tag, short in MODELS.items():
            fs = [f for r in RUNS for f in [os.path.join(r, f"{t}_1000_{tag}_outputs", f"completions_{short}.json")]
                  if os.path.exists(f)]
            if not fs:
                continue
            f = fs[0]  # DS-RL's part-2 completions are the same file as full/probe
            for c in json.load(open(f)):
                i = int(c["index"])
                question, label = q[t].question[i], int(q[t].answer[i])
                e, a = exact_value(question), num(c.get("generated_answer"))
                ok = a is not None and is_correct_fixed(a, label, e, label_is_reading(question))
                d = None if a is None else abs(F(a) - e)
                yield dict(template=t, model=short, exact_value=float(e), right=ok,
                           exact=a is not None and e.denominator == 1 and a == e,
                           tol1=published_tol1(c.get("generated_answer"), label),
                           off_le10=(not ok) and d is not None and d <= 10,
                           off_gt10=(not ok) and d is not None and d > 10, none=a is None,
                           truncated=c.get("finish_reason") == "length")


def main():
    df = pd.DataFrame(rows())
    cols = ["right", "exact", "tol1", "off_le10", "off_gt10", "none", "truncated"]
    tab = df.groupby(["template", "model"])[cols].sum().astype(int)
    tab.insert(0, "n", df.groupby(["template", "model"]).size())
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    tab.to_csv(os.path.join(HERE, "out", "answer_accuracy.csv"))
    df["size_q"] = df.groupby("template").exact_value.transform(lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    size = df.pivot_table(index=["template", "model"], columns="size_q", values="right", aggfunc="mean", observed=False)
    rng = df.groupby("template").exact_value.agg(["min", "max"]).round(0)
    size.to_csv(os.path.join(HERE, "out", "answer_accuracy_by_size.csv"))
    pd.set_option("display.width", 200)
    print(tab.to_string())
    print("\nShare right by answer-size quartile within each template (Q1 smallest):")
    print(size.round(3).to_string())
    print("\nExact-value range per template:")
    print(rng.to_string())


if __name__ == "__main__":
    main()
