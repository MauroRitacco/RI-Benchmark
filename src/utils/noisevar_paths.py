"""Path helpers for the noisevar dataset (mode RI_BENCH_MODE=noisevar).

Inputs live at data/noisevar/noisevar/<snr<L>>/<target>/ and outputs under
experiments/noisevar/<snr<L>>/<target>/, with the level directory derived
from the target's '_snr<L>' name suffix so target names stay plain filenames
everywhere (sweep state CSV, cell ids, logs).

The GT is FIXED per target (the varying quantity is the noise), so it lives
OUTSIDE the cell: the cell .json carries 'gt_fits', an absolute path to the
original unscaled FITS in data/simulated/<target>/. Metrics must use it
as-is, never rescaled.
"""
import json
import os
import re

_SNRSUFFIX_RE = re.compile(r"_snr(\d+)$")


def noisevar_level_dir(target):
    """'disk_..._snr50' -> 'snr50'; '' if the name has no '_snr<L>' suffix."""
    m = _SNRSUFFIX_RE.search(target)
    return f"snr{m.group(1)}" if m else ""


def noisevar_dirs(base_dir, target, kind):
    """(inputs_dir, outputs_dir) for a noisevar target.

    kind: 'data' -> data/noisevar/noisevar, 'exp' -> experiments/noisevar.
    """
    lvl = noisevar_level_dir(target)
    if kind == "data":
        root = os.path.join(base_dir, "data", "noisevar", "noisevar")
    else:
        root = os.path.join(base_dir, "experiments", "noisevar")
    if lvl:
        root = os.path.join(root, lvl)
    return root


def noisevar_gt_path(base_dir, target):
    """Absolute GT FITS path from the cell .json 'gt_fits', or None.

    Points to the original unscaled GT (data/simulated/<target>/); used
    as-is for metrics and headers, never rescaled.
    """
    jpath = os.path.join(noisevar_dirs(base_dir, target, "data"), target,
                         f"{target}.json")
    try:
        with open(jpath) as f:
            p = json.load(f).get("gt_fits", "")
        return p if os.path.isfile(p) else None
    except Exception:
        return None
