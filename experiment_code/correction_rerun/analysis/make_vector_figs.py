"""Run a figure script unchanged, and save each figure as a PNG and a PDF, as in the paper.

The PDFs use TrueType fonts instead of matplotlib's default Type 3 fonts. Regression bands
use seed 0, so they are the same in every run.

Usage: python3 make_vector_figs.py make_probe_figs_rerun.py
       python3 make_vector_figs.py make_fig2_rerun.py
"""
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure as mf  # noqa: E402
import seaborn as sns  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42

script = os.path.abspath(sys.argv[1])
orig_save = mf.Figure.savefig
orig_reg = sns.regplot


def save(self, fname, *a, **k):
    orig_save(self, fname, *a, **k)
    if str(fname).endswith(".png"):
        orig_save(self, str(fname)[:-4] + ".pdf", *a, **{**k, "format": "pdf"})


mf.Figure.savefig = save
sns.regplot = lambda *a, **k: orig_reg(*a, **{"seed": 0, **k})
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
