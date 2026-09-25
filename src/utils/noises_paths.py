"""Path helpers for the noises dataset (mode RI_BENCH_MODE=noises).

Inputs live at data/simulated/noises/<L>s/<target>/ and outputs under
experiments/simulated/noises/<L>s/<target>/, with the level directory derived
from the target's '_snr<L>' name suffix so target names stay plain filenames
everywhere (sweep state CSV, cell ids, logs).
"""
import os
import re

_SNRSUFFIX_RE = re.compile(r"_snr(\d+)$")


def snr_level_dir(target):
    """'disk_..._snr50' -> '50s'; '' if the name has no '_snr<L>' suffix."""
    m = _SNRSUFFIX_RE.search(target)
    return f"{m.group(1)}s" if m else ""


def noises_dirs(base_dir, target, kind):
    """(inputs_dir, outputs_dir) for a noises target.

    kind: 'data' -> data/simulated/noises, 'exp' -> experiments/simulated/noises.
    """
    lvl = snr_level_dir(target)
    root = os.path.join(base_dir, "data" if kind == "data" else "experiments",
                        "simulated", "noises")
    if lvl:
        root = os.path.join(root, lvl)
    return root
