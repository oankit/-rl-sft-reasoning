"""Layer-wise probe evaluation for the correction rerun.

Same probe as the paper (LogisticRegressionCV, L2, lbfgs, C grid {0.01..100},
5-fold CV, class_weight='balanced'), with these fixes:
- stratified 70/15/15 split, repeated over several split seeds;
- reports balanced accuracy and AUROC next to accuracy and the majority baseline;
- marks runs whose test split has too few negatives or positives to evaluate;
- --balance subsamples equal numbers of correct and incorrect examples, as the paper's
  method section describes (the released code never did): per split seed, all of the
  minority class and an equal random sample of the majority class, then the split.

The *_sd columns are the sample SD (ddof=1) over the split seeds. The splits
overlap, so this is split variability, not a confidence interval.
--cv-scoring picks the score LogisticRegressionCV uses to choose C. The paper used
the default, accuracy. --labels exact recomputes labels from completions_<model>.json
with an exact match; the default keeps the generation labels (within 1 of the truth).

Input: an extraction output dir with activations_<model>.npy [L, N, D],
labels_<model>.npy [N] and metadata_<model>.json (from extract_activations.py).
Output: one CSV row per (layer, seed) plus a per-layer summary CSV.

Usage: python probe_eval.py <extract_dir> <model_short> <out_csv_prefix> [--seeds 5] [--jobs -1]
           [--cv-scoring accuracy|balanced_accuracy|roc_auc] [--labels tol1|exact] [--balance]
       python probe_eval.py --selfcheck
"""
import argparse
import csv
import json
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_CLASS_TEST = 10  # below this many test examples in either class, flag as not evaluable


def split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return tr, va, te


def balanced_split(y, seed):
    """Split of a 50/50 subsample: every minority example plus an equal draw of the majority."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n = min(len(pos), len(neg))
    idx = np.sort(np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]))
    tr, va, te = split(y[idx], seed)
    return idx[tr], idx[va], idx[te]


def fit_eval(X, y, tr, te, seed, scoring="accuracy", scale=False):
    clf = LogisticRegressionCV(Cs=C_GRID, cv=5, penalty="l2", solver="lbfgs", max_iter=5000,
                               class_weight="balanced", random_state=seed, n_jobs=1,
                               scoring=scoring)
    # --scale standardizes features on the training split only. Unscaled raw activations
    # (the paper's setup) hit max_iter and take over 10 min per fit.
    if scale:
        clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    yt = y[te]
    both = len(np.unique(yt)) == 2
    return dict(acc=accuracy_score(yt, pred),
                bal_acc=balanced_accuracy_score(yt, pred) if both else float("nan"),
                auroc=roc_auc_score(yt, p) if both else float("nan"),
                majority=max(yt.mean(), 1 - yt.mean()), best_C=float(clf[-1].C_[0] if scale else clf.C_[0]))


def exact_labels(extract_dir, model, y_saved):
    """Labels with an exact match, in the row order of extract_activations.py."""
    comps = json.load(open(os.path.join(extract_dir, f"completions_{model}.json")))
    valid = [c for c in comps if c.get("generated_answer") and c["generated_answer"] != "nan"]
    tol1, exact = [], []
    for c in valid:
        try:
            g = float(c["generated_answer"])
        except ValueError:
            g = float("nan")
        tol1.append(abs(g - c["ground_truth"]) <= 1)
        exact.append(g == c["ground_truth"])
    # The tolerance-1 recomputation must reproduce the saved labels, or the rows do not line up.
    assert np.array_equal(np.array(tol1, dtype=int), y_saved), "completions do not match labels"
    return np.array(exact, dtype=int)


def subset_mask(extract_dir, model, meta, subset):
    """Rows to keep: 'finished' drops truncated outputs, 'boxed' drops outputs without \\boxed{."""
    if subset == "boxed":
        return np.array(meta["boxed_found"], dtype=bool)
    comps = json.load(open(os.path.join(extract_dir, f"completions_{model}.json")))
    return np.array([comps[i].get("finish_reason") != "length" for i in meta["valid_indices"]], dtype=bool)


def run(extract_dir, model, out_prefix, seeds, jobs, scoring="accuracy", label_mode="tol1", subset="all", scale=False,
        balance=False):
    acts = np.load(os.path.join(extract_dir, f"activations_{model}.npy"), mmap_mode="r")
    y = np.load(os.path.join(extract_dir, f"labels_{model}.npy")).astype(int)
    meta = json.load(open(os.path.join(extract_dir, f"metadata_{model}.json")))
    if label_mode == "exact":
        y_tol = y
        y = exact_labels(extract_dir, model, y)
        print(f"{model}: exact labels relabel {int((y != y_tol).sum())} of {len(y)} examples")
    keep = np.ones(len(y), dtype=bool) if subset == "all" else subset_mask(extract_dir, model, meta, subset)
    assert len(keep) == len(y)
    y = y[keep]
    print(f"{model}: subset={subset} keeps {int(keep.sum())} of {len(keep)} examples")
    L = acts.shape[0]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if min(n_pos, n_neg) == 0:  # one class only: stratified split works, but the fit crashes
        print(f"{model}: {n_pos} correct / {n_neg} incorrect, not trainable (one class only)")
        return None
    try:
        splits = [(balanced_split if balance else split)(y, s) for s in range(seeds)]
    except ValueError as e:  # a class too small for two stratified splits
        print(f"{model}: {n_pos} correct / {n_neg} incorrect, not trainable ({e})")
        return None
    jobsl = [(l, s) for l in range(L) for s in range(seeds)]
    res = Parallel(n_jobs=jobs, verbose=5)(
        delayed(fit_eval)(np.asarray(acts[l], dtype=np.float32)[keep], y, splits[s][0], splits[s][2], s,
                          scoring, scale)
        for l, s in jobsl)
    rows = []
    for (l, s), r in zip(jobsl, res):
        te_y = y[splits[s][2]]
        rows.append(dict(layer=l, seed=s, labels=label_mode, cv_scoring=scoring, subset=subset, scaled=scale,
                         balanced=balance, n=len(y), n_used=sum(map(len, splits[s])), n_correct=n_pos, n_incorrect=n_neg,
                         test_pos=int(te_y.sum()), test_neg=int(len(te_y) - te_y.sum()),
                         evaluable=bool(min(te_y.sum(), len(te_y) - te_y.sum()) >= MIN_CLASS_TEST),
                         is_last_layer_postnorm=(l == L - 1), **r))
    with open(out_prefix + "_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = []
    for l in range(L):
        rs = [r for r in rows if r["layer"] == l]
        s = dict(layer=l, labels=label_mode, cv_scoring=scoring, subset=subset, balanced=balance,
                 n_used=rs[0]["n_used"], evaluable=all(r["evaluable"] for r in rs), is_last_layer_postnorm=(l == L - 1),
                 n_correct=n_pos, n_incorrect=n_neg)
        for k in ("acc", "bal_acc", "auroc", "majority"):
            v = np.array([r[k] for r in rs], dtype=float)
            s[k + "_mean"] = float(np.nanmean(v))
            s[k + "_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
        summ.append(s)
    with open(out_prefix + "_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(json.dumps(dict(model=model, n_correct=n_pos, n_incorrect=n_neg,
                          boxed_found_rate_correct=meta.get("boxed_found_rate_correct"),
                          boxed_found_rate_incorrect=meta.get("boxed_found_rate_incorrect"),
                          # Pre-norm layers only, and only layers evaluable on every seed.
                          best_layer_bal_acc=max((s["bal_acc_mean"] for s in summ
                                                  if s["evaluable"] and not s["is_last_layer_postnorm"]),
                                                 default=None))))
    return summ


def selfcheck():
    # A planted linear signal must be found; pure noise must sit near chance.
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    X_sig = rng.normal(size=(400, 32)); X_sig[:, 0] += 3 * y
    X_noise = rng.normal(size=(400, 32))
    tr, _, te = split(y, 0)
    assert fit_eval(X_sig, y, tr, te, 0)["bal_acc"] > 0.85
    assert abs(fit_eval(X_noise, y, tr, te, 0)["bal_acc"] - 0.5) < 0.15
    assert fit_eval(X_sig, y, tr, te, 0, "balanced_accuracy")["bal_acc"] > 0.85
    assert fit_eval(X_sig, y, tr, te, 0, scale=True)["bal_acc"] > 0.85
    assert abs(fit_eval(X_noise, y, tr, te, 0, scale=True)["bal_acc"] - 0.5) < 0.15

    # End to end on a fake extraction dir, including a 5-negative cell that cannot be split.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        acts = np.stack([X_noise, X_sig, X_sig])  # [L=3, N, D]; layer 2 plays the post-norm layer
        np.save(os.path.join(d, "activations_m.npy"), acts)
        np.save(os.path.join(d, "labels_m.npy"), y.astype(np.int8))
        json.dump({}, open(os.path.join(d, "metadata_m.json"), "w"))
        # Wrong answers are off by 2; every 10th correct one is off by 1 (correct within tol 1).
        comps = [dict(generated_answer=str(10 + (0 if yi else 2) + (yi and j % 10 == 0)),
                      ground_truth=10.0) for j, yi in enumerate(y)]
        json.dump(comps, open(os.path.join(d, "completions_m.json"), "w"))
        assert (exact_labels(d, "m", y) != y).sum() == sum(yi and j % 10 == 0 for j, yi in enumerate(y)) > 0
        summ = run(d, "m", os.path.join(d, "out"), 2, 1, label_mode="exact")
        assert len(summ) == 3 and summ[2]["is_last_layer_postnorm"] and summ[1]["bal_acc_mean"] > 0.85
        # Subsets: every 4th row truncated, every 5th row unboxed.
        for c, j in zip(comps, range(len(comps))):
            c["finish_reason"] = "length" if j % 4 == 0 else "stop"
        json.dump(comps, open(os.path.join(d, "completions_m.json"), "w"))
        json.dump(dict(valid_indices=list(range(len(y))), boxed_found=[j % 5 != 0 for j in range(len(y))]),
                  open(os.path.join(d, "metadata_m.json"), "w"))
        m = json.load(open(os.path.join(d, "metadata_m.json")))
        assert subset_mask(d, "m", m, "finished").sum() == len(y) - len(range(0, len(y), 4))
        assert subset_mask(d, "m", m, "boxed").sum() == len(y) - len(range(0, len(y), 5))
        assert run(d, "m", os.path.join(d, "out"), 2, 1, subset="finished")[1]["bal_acc_mean"] > 0.85
        # Balanced: equal classes in every part of the split, majority baseline 0.5.
        for sd in range(3):
            parts = balanced_split(y, sd)
            n_min = min(int(y.sum()), int(len(y) - y.sum()))
            assert sum(map(len, parts)) == 2 * n_min and len(set(np.concatenate(parts))) == 2 * n_min
            assert all(abs(int(y[p].sum()) * 2 - len(p)) <= 1 for p in parts)
        bal = run(d, "m", os.path.join(d, "out"), 2, 1, subset="finished", balance=True)
        assert bal[1]["bal_acc_mean"] > 0.85 and bal[1]["majority_mean"] <= 13 / 25  # 162 balanced rows: 25 test rows, 13 vs 12
        y5 = np.ones(1000, dtype=np.int8); y5[:5] = 0
        np.save(os.path.join(d, "labels_m.npy"), y5)
        np.save(os.path.join(d, "activations_m.npy"), rng.normal(size=(1, 1000, 4)))
        assert run(d, "m", os.path.join(d, "out"), 2, 1) is None
        np.save(os.path.join(d, "labels_m.npy"), np.ones(1000, dtype=np.int8))  # no negatives
        assert run(d, "m", os.path.join(d, "out"), 2, 1) is None
    print("selfcheck ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("extract_dir", nargs="?")
    p.add_argument("model", nargs="?")
    p.add_argument("out_prefix", nargs="?")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--cv-scoring", default="accuracy",
                   choices=["accuracy", "balanced_accuracy", "roc_auc"])
    p.add_argument("--labels", default="tol1", choices=["tol1", "exact"])
    p.add_argument("--scale", action="store_true")
    p.add_argument("--subset", default="all", choices=["all", "finished", "boxed"])
    p.add_argument("--balance", action="store_true", help="50/50 subsample per split seed")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    if a.selfcheck or not a.extract_dir:
        selfcheck()
    else:
        run(a.extract_dir, a.model, a.out_prefix, a.seeds, a.jobs, a.cv_scoring, a.labels, a.subset, a.scale,
            a.balance)
