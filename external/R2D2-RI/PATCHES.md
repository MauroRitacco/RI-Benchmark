# Local patches on the vendored code

This directory is a vendored copy of [basp-group/R2D2-RI](https://github.com/basp-group/R2D2-RI)
(snapshotted at upstream commit `2266925`, "updated for natural weighting"),
with minimal local modifications required by the benchmark. This file
documents each patch.

## 1. `src/model/inference.py` — single-layer R2D2: do not add the previous estimate

- **Symptom**: multi-iteration R2D2 reconstructions (`layers: 1`,
  `num_iter > 1`) with a data range much larger than the ground truth
  `[0, 1]` (observed: `[0, 3.8893]`).
- **Cause**: training (`src/model/R2D2_model.py::R2D2_forward`) trains the
  single-layer U-Net to predict the full normalized image, but inference
  (`src/model/inference.py::forward`) additionally added the previous
  estimate (`+ output_n`) in the `layers == 1` branch. Every iteration
  accumulated the previous output on top of the network prediction.
- **Change**: the `+ output_n` skip connection was removed in the
  `layers == 1` branch of `forward()` (with an explanatory comment in the
  code).
- **Scope**: only affects `layers == 1`. The `layers > 1` branch (R3D3)
  keeps the skip connection, because there the training does use residual
  learning.
- **Relation to the reported results**: the patch has been applied in both
  copies (`R2D2/R2D2_small` and this one) since 2026-07-13. The benchmark
  reference reconstructions (`experiments/`, September–October 2026) were
  produced afterwards, with the patch active. Local verification
  (2026-11-02, `../verification/r2d2_inference_fix/`, not part of this
  repository): re-running inference with this copy reproduces the test
  reference reconstruction (`disk_s0p5_r1_c0_x0_000`, winning configuration
  `k5`) with a maximum difference of 4.4e-16, and the metrics match
  `results/test/test_state.csv` (PSNR 28.7372 vs 28.74, SSIM 0.981365 vs
  0.9814, MSE 0.001337 vs 0.0013).
- **Not confirmed**: it has not been verified that *all* historical runs
  (e.g. the older August sweep runs) used exactly this copy of the file; the
  `R2D2_small` CHANGELOG records the fix as "[Unreleased]".

## 2. `src/utils/args.py` — CLI `--num_chans` and override order

- **Change**: added the `--num_chans` argument and fixed the merge of CLI
  overrides over the YAML in `parse_args_imaging()` (previously
  `ImagingArgs` was built before applying the overrides, ignoring them).
- **Motivation**: the benchmark runner (`scripts/run_r2d2.sh`) passes
  `--num_chans` and expects CLI arguments to take priority over the YAML.

## 3. Submodule `src/ri_measurement_operator` — UV index clamp

- **File**: `pysrc/utils/gen_imaging_weights.py` (submodule in "dirty"
  state, 3 lines).
- **Change**: added a `torch.clamp` of the grid indices `p` and `q` to the
  valid range before `scatter_add_`, with the comment: "Clamp indices to
  valid grid range (UV points at boundaries can produce out-of-bounds
  indices)".
- **Motivation**: UV points at the grid boundaries could generate
  out-of-range indices in the imaging-weights computation (Briggs).
