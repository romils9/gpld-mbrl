# GPLD-MBRL

GPLD-MBRL is a research codebase for DreamerV3 with gradient-penalty Lipschitz
regularization. It keeps the original `dreamerv3` and `embodied` Python package
names while publishing the project as `gpld-mbrl`.

This initial release focuses on GPLD-DreamerV3; future releases may include
additional model-based RL baselines and GPLD variants.

The main addition is a configurable gradient penalty for RSSM latent distributions. The paper’s default GPLD-DreamerV3 setting applies the penalty to the posterior latent probability map, while the code also exposes prior and joint prior-posterior variants for ablations.

## Installation

This code has been prepared for Python 3.11.

```bash
conda create -n gpld-mbrl python=3.11 -y
conda activate gpld-mbrl
pip install -U pip setuptools wheel
pip install -e .
```

The default install uses CPU-compatible JAX. For CUDA runs, install the JAX
wheel that matches your CUDA stack before or after installing this package. For
example:

```bash
pip install -U "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

DeepMind Control tasks require MuJoCo/EGL system support on headless machines.
Set `MUJOCO_GL=egl` when running on GPU servers without a display.

## Quick Checks

```bash
python -m compileall dreamerv3 embodied
python - <<'PY'
import dreamerv3
import embodied
print("imports ok")
PY
```

A short CPU debug run can be launched with:

```bash
python -m dreamerv3.main \
  --logdir ./logdir/debug/{timestamp} \
  --configs dmc_proprio debug \
  --task dmc_walker_walk \
  --run.steps 1000 \
  --run.envs 1 \
  --run.eval_envs 1 \
  --jax.platform cpu \
  --jax.prealloc False
```

## Training Examples

Proprioceptive DMC with posterior regularization:

```bash
POST_LAM=0.5 PRIOR_LAM=0.0 ENV_NAME=dmc_walker_walk \
SAMPLE_FRAC=0.5 STEPS=1000000 TRAIN_RATIO=512 \
./run_dmc_lipschitz_seeds.sh
```

Pixel DMC with posterior regularization:

```bash
POST_LAM=0.5 PRIOR_LAM=0.0 ENV_NAME=dmc_cheetah_run \
SAMPLE_FRAC=0.5 STEPS=1000000 TRAIN_RATIO=512 \
./run_dmc_pixel_lipschitz_seeds.sh
```

Most experiment settings can be overridden via environment variables. See the
scripts and `dreamerv3/configs.yaml` for the full configuration surface.

## Main configuration knobs

The GPLD experiments can be configured through environment variables passed to the run scripts.

| Paper quantity | Script environment variable | Dreamer config key | Typical value |
|---|---:|---:|---:|
| Posterior penalty coefficient \(\lambda^{\mathrm{post}}_0\) | `POST_LAM` | `agent.dyn.rssm.gp.post_lam` | `0.5` |
| Prior penalty coefficient \(\lambda^{\mathrm{prior}}_0\) | `PRIOR_LAM` | `agent.dyn.rssm.gp.prior_lam` | `0.0` |
| Sampling fraction \(\rho\) | `SAMPLE_FRAC` | `agent.dyn.rssm.gp.sample_frac` | `0.5` |
| GP stochastic rows | `GP_ROWS` | `agent.dyn.rssm.gp.gp_rows` | `32` |
| Random GP row sampling | `GP_ROWS_RANDOM` | `agent.dyn.rssm.gp.gp_rows_random` | `false` |
| Decay type | `DECAY_TYPE` | `agent.dyn.rssm.gp.decay` | `sqrt_time` |
| Minimum decayed coefficient | `MIN_LAM` | `agent.dyn.rssm.gp.min_lam` | `0.001` |
| GP start step | `START_STEP` | `agent.dyn.rssm.gp.start_step` | `0` |
| DMC task | `ENV_NAME` | `task` | `dmc_walker_walk` |
| Training steps | `STEPS` | `run.steps` | `1000000` |
| Train ratio | `TRAIN_RATIO` | `run.train_ratio` | `512` |
| Seed list | `SEEDS` | `seed` per run | `"0 1 2 3 4"` |
| Log directory root | `LOGDIR_ROOT` | `logdir` prefix | `./logdir/proprio` |

For example:

```bash
SEEDS="0 1 2 3 4" ENV_NAME=dmc_walker_walk STEPS=1000000 \
PRIOR_LAM=0.0 POST_LAM=0.5 SAMPLE_FRAC=0.5 DECAY_TYPE=sqrt_time \
LOGDIR_ROOT=./logdir/proprio ./run_dmc_lipschitz_seeds.sh
```

## Reproduction

See `REPRODUCTION.md` for environment setup, training, evaluation, score
processing, smoothness analysis, and fresh-clone validation notes.

## Citation

If you use this repository, please cite our GPLD paper:

```bibtex
@article{sonigra2026gpld,
  title={Dreaming Smoothly and Sample Efficiently with Gradient Penalized Latent Dynamics},
  author={Sonigra, Romil V. and Kumar, P. R.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

This repository builds on DreamerV3. Please also cite the original DreamerV3 paper:

```bibtex
@article{hafner2025dreamer,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  volume={640},
  pages={647--653},
  year={2025},
  doi={10.1038/s41586-025-08744-2}
}
```

## License

The code is released under the MIT License. See `LICENSE` and `NOTICE`.
