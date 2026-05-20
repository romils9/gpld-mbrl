# Reproduction Guide

This guide describes how to reproduce GPLD-MBRL runs from a clean clone.

## Environment

```bash
git clone git@github.com:romils9/gpld-mbrl.git
cd gpld-mbrl
conda create -n gpld-mbrl python=3.11 -y
conda activate gpld-mbrl
pip install -U pip setuptools wheel
pip install -e .
```

For CUDA runs, install the matching JAX CUDA wheel:

```bash
pip install -U "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export MUJOCO_GL=egl
```

## Training

The public scripts default to local, relative output directories. Override the
parameters with environment variables.

Proprioceptive DMC:

```bash
SEEDS="0 1 2 3 4" ENV_NAME=dmc_walker_walk STEPS=100000 \
PRIOR_LAM=0.0 POST_LAM=0.5 DECAY_TYPE=sqrt_time \
LOGDIR_ROOT=./logdir/proprio ./run_dmc_lipschitz_seeds.sh
```

Pixel DMC:

```bash
SEEDS="0 1 2 3 4" ENV_NAME=dmc_cheetah_run STEPS=100000 \
PRIOR_LAM=0.0 POST_LAM=0.5 DECAY_TYPE=sqrt_time \
LOGDIR_ROOT=./logdir/vision ./run_dmc_pixel_lipschitz_seeds.sh
```

To initialize from a checkpoint, pass the path explicitly:

```bash
PRETRAINED_CHECKPOINT=/path/to/ckpt/step_00250000_step \
PRETRAINED_CHECKPOINT_REGEX='^(enc|dec)/' \
./run_dmc_lipschitz_seeds.sh
```

## Evaluation

Evaluate checkpoint folders with:

```bash
CHECKPOINT_BASE=./logdir/proprio \
EVAL_OUTPUT_DIR=./evals/proprio \
TARGET_CKPT_STEP=100000 \
./run_eval_proprio_ckpts.sh
```

Use `LIST_ONLY=true` to inspect what would be evaluated before launching jobs.

## Processing Scores

Process evaluation JSONL files into per-seed and aggregate arrays:

```bash
python process_evals.py \
  --input ./evals/proprio \
  --output ./processed_scores/proprio
```

Generate grid plots from processed NPZ files:

```bash
python plot_grid_individuals.py --processed-root ./processed_scores/proprio --output ./plots
python plot_grid_xmax.py --processed-root ./processed_scores/proprio --output ./plots
```

The plotting scripts expect the aggregate filenames produced by
`process_evals.py`.

## Smoothness Analysis

Run perturbation smoothness analysis against an experiment directory that
contains a `ckpt/` subdirectory:

```bash
python analyze_smoothness_2.py \
  --checkpoint_dir ./logdir/proprio/example_run \
  --all_checkpoints \
  --skip_steps 5
```

Compare saved smoothness outputs with:

```bash
python compare_lipschitz.py \
  --baseline_dir /path/to/baseline_run \
  --lipschitz_dir /path/to/gpld_run
```

## Reproducibility Notes

The run scripts set `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, and
`TF_CUDNN_DETERMINISTIC` for each seed. Exact reproducibility can still vary
with GPU model, driver, CUDA/JAX versions, and simulator backend.

Recommended metadata to record for each run:

- command line and environment overrides
- seed list
- GPU model and driver
- Python, JAX, CUDA, and MuJoCo versions
- git commit SHA

## Fresh-Clone Validation

After publishing, validate the repository from outside the working tree:

```bash
rm -rf /tmp/gpld-mbrl-fresh
git clone git@github.com:romils9/gpld-mbrl.git /tmp/gpld-mbrl-fresh
cd /tmp/gpld-mbrl-fresh
conda create -n gpld-mbrl-test python=3.11 -y
conda activate gpld-mbrl-test
pip install -U pip setuptools wheel
pip install -e .
python -m compileall dreamerv3 embodied
python - <<'PY'
import dreamerv3
import embodied
import plotting_neurips
print("fresh clone imports ok")
PY
```
