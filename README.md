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
STEPS=100000 TRAIN_RATIO=512 ./run_dmc_lipschitz_seeds.sh
```

Pixel DMC with posterior regularization:

```bash
POST_LAM=0.5 PRIOR_LAM=0.0 ENV_NAME=dmc_cheetah_run \
STEPS=100000 TRAIN_RATIO=512 ./run_dmc_pixel_lipschitz_seeds.sh
```

Most experiment settings can be overridden via environment variables. See the
scripts and `dreamerv3/configs.yaml` for the full configuration surface.

## Reproduction

See `REPRODUCTION.md` for environment setup, training, evaluation, score
processing, smoothness analysis, and fresh-clone validation notes.

## Attribution

This repository includes and modifies DreamerV3. Please cite the original
DreamerV3 work when using this code:

```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

The code is released under the MIT License. See `LICENSE` and `NOTICE`.
