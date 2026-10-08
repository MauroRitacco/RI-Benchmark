# RI-Benchmark

Benchmark of radio-interferometric image reconstruction on simulated protoplanetary
disks, with known ground truth. Four methods are compared on identical visibilities:
CLEAN (CASA `tclean`), WSClean, FISTA with a wavelet prior (DeepInverse), and R2D2
(iterative deep-learning reconstruction).

## Main results

Held-out test set (n=360 disks), winning configuration per method, averaged over
targets. Configurations were selected by a hyper-parameter sweep of 11
configurations (CASA: 4, WSClean: 3, DeepInverse: 3, R2D2: 1).
Full report: `results/test/test_results.txt`.

| Method       | Config        | PSNR (dB) | SSIM   | Time/img (s) |
|--------------|---------------|-----------|--------|--------------|
| DeepInverse  | `l0p1_i50`    | **46.48** | **0.9876** | 13.89    |
| R2D2         | `k5`          | 37.72     | 0.9494 | 1.48         |
| CASA (CLEAN) | `g0p1_t1p0`   | 29.77     | 0.8192 | 3.47         |
| WSClean      | `g0p1_t1p0`   | 29.26     | 0.8034 | 1.11         |

R2D2 runs ~1.5 s/image, about 10x faster than DeepInverse, at a cost of ~9 dB PSNR.
Hyper-parameter sweep grids: `results/sweep/`.

## Repository structure

- `scripts/` — entry points: `run_{casa,wsclean,deepinv,r2d2}.sh`, `run_all.sh`, `run_sweep.sh`, sweep/test target and grid files.
- `src/` — method implementations (`algorithms/`), sweep engine (`sweep/`), metrics (`compute_metrics.py`).
- `config/imaging/` — R2D2 imager configs used by the benchmark.
- `requirements/` — one pinned `requirements.txt` per environment.
- `data/` — targets: `data/{simulated,simulated/test,archival}/<target>/` with `.MS`, `.mat`, ground truth and a `<target>.yaml` of imaging parameters. Not tracked.
- `experiments/` — reconstruction outputs. Not tracked.
- `results/` — metric tables and sweep/test reports.
- `external/` — vendored [basp-group/R2D2-RI](https://github.com/basp-group/R2D2-RI) used for R2D2 inference. Not tracked.
- `.envs/` — one virtualenv per algorithm. Not tracked.

## Reproducing

Requirements: Python 3.10, pip, and the WSClean system binary.

```bash
./scripts/create_envs.sh          # create .envs/* (venv per method)
# install each env from requirements/*.txt, then fetch the R2D2 checkpoints:
hf download MauroRitacco/R2Disk2 --include "ckpt/*" --local-dir external/R2D2-RI

./scripts/run_sweep.sh                       # hyper-parameter sweep (scripts/sweep_targets.txt)
./scripts/run_sweep.sh --test                # held-out test set -> results/test/
./scripts/run_all.sh --target <name>         # single target, all four methods
```


## Data

Synthetic disks and visibilities
are generated with the companion [DiskGenerator](https://github.com/mauroritacco/DiskGenerator)
repository; the test set is a held-out split of the same generator (see its
`sweep_experiment/` grids). Archival Measurement Sets convert to `.mat` with
`src/utils/ms2mat.py`.

## License

GPL-3.0. See [LICENSE](LICENSE).
