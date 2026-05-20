#!/usr/bin/env bash
set -euo pipefail

# DreamerV3 with Lipschitz Smoothness Constraints — DMC proprio across 5 seeds (0..4)
# Usage: ./run_dmc_lipschitz_seeds.sh
# Optional: override parameters via env vars before the call:
#   PRIOR_LAM=0.01 POST_LAM=0.01 DECAY_TYPE=step ./run_dmc_lipschitz_seeds.sh
#   LOGDIR_ROOT=./logdir/proprio SEEDS="0 1 2 3 4" ./run_dmc_lipschitz_seeds.sh
#
# Decay types:
#   - none: No decay (constant lambda)
#   - linear: Linear decay over DECAY_STEPS
#   - exponential: Exponential decay over DECAY_STEPS
#   - cosine: Cosine annealing over DECAY_STEPS
#   - step: Discrete decay by DECAY_FACTOR every DECAY_INTERVAL steps
#   - sqrt_time: Decay as base_lam / sqrt(t+1) - slower decay than linear
#   - custom: Use SCHEDULE_STEPS and SCHEDULE_VALUES arrays

IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0}"
# SEEDS=(61)

# ============================================================================
# General training parameters (can be overridden by env vars)
# ============================================================================
STEPS="${STEPS:-250500}"
ENVS="${ENVS:-16}"
EVAL_ENVS="${EVAL_ENVS:-4}"
TRAIN_RATIO="${TRAIN_RATIO:-512}"
REPEAT="${REPEAT:-2}"
ENV_NAME="${ENV_NAME:-dmc_walker_run}"
LOGDIR_ROOT="${LOGDIR_ROOT:-./logdir/gpld_proprio}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
PRETRAINED_CHECKPOINT_REGEX="${PRETRAINED_CHECKPOINT_REGEX:-}"

# ============================================================================
# Lipschitz constraint parameters (gradient penalty)
# ============================================================================
PRIOR_LAM="${PRIOR_LAM:-0.5}"          # Prior gradient penalty weight
POST_LAM="${POST_LAM:-0.0}"            # Posterior gradient penalty weight
SAMPLE_FRAC="${SAMPLE_FRAC:-0.5}"        # Fraction of samples for GP computation
GP_ROWS="${GP_ROWS:-32}"                 # Number of stoch rows to penalize (1-32, 32=all)
GP_ROWS_RANDOM="${GP_ROWS_RANDOM:-False}" # Randomly sample rows each step instead of first-N

# ============================================================================
# Decay type: none, linear, exponential, cosine, step, custom, sqrt_time
# ============================================================================
DECAY_TYPE="${DECAY_TYPE:-sqrt_time}"

# Parameters for linear/exponential/cosine decay
DECAY_STEPS="${DECAY_STEPS:-200000}"     # Steps over which to decay
MIN_LAM="${MIN_LAM:-0.1}"                # Minimum lambda value (floor)

# Parameters for step decay
DECAY_INTERVAL="${DECAY_INTERVAL:-5000}" # Decay every N steps
DECAY_FACTOR="${DECAY_FACTOR:-0.9}"      # Multiply by this factor each interval

# Parameters for custom decay (space-separated lists)
SCHEDULE_STEPS="${SCHEDULE_STEPS:-100000 200000 300000 400000 500000}"     # e.g., "5000 10000 25000 50000"
SCHEDULE_VALUES="${SCHEDULE_VALUES:-0.0007 0.0003 0.0001 0.00005 0.0}"   # e.g., "0.08 0.05 0.02 0.01"

# ============================================================================
# Build experiment name
# ============================================================================
if awk -v a="$PRIOR_LAM" -v b="$POST_LAM" 'BEGIN{exit !((a+0)>0 || (b+0)>0)}'; then
  EXP_SUFFIX="lipschitz_prior${PRIOR_LAM}_post${POST_LAM}_rows${GP_ROWS}"
  if [[ "$GP_ROWS_RANDOM" == "True" ]]; then
    EXP_SUFFIX="${EXP_SUFFIX}rnd"
  fi
  if [[ "$DECAY_TYPE" != "none" ]]; then
    EXP_SUFFIX="${EXP_SUFFIX}_${DECAY_TYPE}"
  fi
else
  EXP_SUFFIX="baseline"
fi

# ============================================================================
# Build decay-specific arguments based on DECAY_TYPE
# ============================================================================
build_decay_args() {
  local decay_args=""

  case "$DECAY_TYPE" in
    none)
      # No decay - just set decay type
      decay_args="--agent.dyn.rssm.gp.decay none"
      ;;

    linear|exponential|cosine)
      # Continuous decay schedules: need decay_steps and min_lam
      decay_args="--agent.dyn.rssm.gp.decay ${DECAY_TYPE}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.decay_steps ${DECAY_STEPS}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.min_lam ${MIN_LAM}"
      ;;

    step)
      # Step-based discrete decay: need decay_interval, decay_factor, min_lam
      decay_args="--agent.dyn.rssm.gp.decay step"
      decay_args="${decay_args} --agent.dyn.rssm.gp.decay_interval ${DECAY_INTERVAL}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.decay_factor ${DECAY_FACTOR}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.min_lam ${MIN_LAM}"
      ;;

    sqrt_time)
      # Square root time decay: lambda = base_lam / sqrt(t + 1)
      # Only needs min_lam as a floor value
      decay_args="--agent.dyn.rssm.gp.decay sqrt_time"
      decay_args="${decay_args} --agent.dyn.rssm.gp.min_lam ${MIN_LAM}"
      ;;

    custom)
      # Custom schedule: need schedule_steps and schedule_values arrays
      if [[ -z "$SCHEDULE_STEPS" ]] || [[ -z "$SCHEDULE_VALUES" ]]; then
        echo "ERROR: DECAY_TYPE=custom requires SCHEDULE_STEPS and SCHEDULE_VALUES"
        echo "Example: SCHEDULE_STEPS='5000 10000 25000' SCHEDULE_VALUES='0.08 0.05 0.02'"
        exit 1
      fi
      decay_args="--agent.dyn.rssm.gp.decay custom"
      decay_args="${decay_args} --agent.dyn.rssm.gp.schedule_steps ${SCHEDULE_STEPS}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.schedule_values ${SCHEDULE_VALUES}"
      decay_args="${decay_args} --agent.dyn.rssm.gp.min_lam ${MIN_LAM}"
      ;;

    *)
      echo "ERROR: Unknown DECAY_TYPE: ${DECAY_TYPE}"
      echo "Valid options: none, linear, exponential, cosine, step, sqrt_time, custom"
      exit 1
      ;;
  esac

  echo "$decay_args"
}

# Get the decay arguments
DECAY_ARGS=$(build_decay_args)

CHECKPOINT_ARGS=()
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  CHECKPOINT_ARGS+=(--run.from_checkpoint "$PRETRAINED_CHECKPOINT")
  if [[ -n "$PRETRAINED_CHECKPOINT_REGEX" ]]; then
    CHECKPOINT_ARGS+=(--run.from_checkpoint_regex "$PRETRAINED_CHECKPOINT_REGEX")
  fi
fi

# ============================================================================
# Print configuration summary
# ============================================================================
echo "=========================================="
echo "Lipschitz DreamerV3 Training Configuration"
echo "=========================================="
echo "Environment: ${ENV_NAME}"
echo "Seeds: ${SEEDS[*]}"
echo "Steps: ${STEPS}"
echo "Train ratio: ${TRAIN_RATIO}"
echo "Logdir root: ${LOGDIR_ROOT}"
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  echo "Pretrained checkpoint: ${PRETRAINED_CHECKPOINT}"
fi
echo ""
echo "Gradient Penalty Settings:"
echo "  Prior lambda: ${PRIOR_LAM}"
echo "  Posterior lambda: ${POST_LAM}"
echo "  Sample fraction: ${SAMPLE_FRAC}"
echo ""
echo "Decay Settings:"
echo "  Decay type: ${DECAY_TYPE}"
case "$DECAY_TYPE" in
  linear|exponential|cosine)
    echo "  Decay steps: ${DECAY_STEPS}"
    echo "  Min lambda: ${MIN_LAM}"
    ;;
  step)
    echo "  Decay interval: ${DECAY_INTERVAL}"
    echo "  Decay factor: ${DECAY_FACTOR}"
    echo "  Min lambda: ${MIN_LAM}"
    ;;
  sqrt_time)
    echo "  Formula: base_lam / sqrt(t + 1)"
    echo "  Min lambda: ${MIN_LAM}"
    ;;
  custom)
    echo "  Schedule steps: ${SCHEDULE_STEPS}"
    echo "  Schedule values: ${SCHEDULE_VALUES}"
    echo "  Min lambda: ${MIN_LAM}"
    ;;
esac
echo "=========================================="
echo ""

# ============================================================================
# Training loop
# ============================================================================
for seed in "${SEEDS[@]}"; do
  echo "=== Starting seed ${seed} with Lipschitz constraints ==="
  echo "    Decay type: ${DECAY_TYPE}"

  # Set environment variables for reproducibility
  export PYTHONHASHSEED="${seed}"
  export TF_DETERMINISTIC_OPS=1
  export TF_CUDNN_DETERMINISTIC=1

  # Run training with decay-specific arguments
  # shellcheck disable=SC2086
  python -m dreamerv3.main \
    --logdir "${LOGDIR_ROOT}/{timestamp}_${ENV_NAME}_${EXP_SUFFIX}_seed${seed}" \
    --script train_eval \
    --configs dmc_proprio size12m \
    --task "${ENV_NAME}" \
    --agent.imag_length 25 \
    --run.envs "${ENVS}" \
    --run.eval_envs "${EVAL_ENVS}" \
    --run.steps "${STEPS}" \
    --run.train_ratio "${TRAIN_RATIO}" \
    --run.save_on_eval False \
    --run.save_every_steps 100000 \
    --run.keep_checkpoints 0 \
    "${CHECKPOINT_ARGS[@]}" \
    --env.dmc.repeat "${REPEAT}" \
    --seed "${seed}" \
    --jax.platform cuda \
    --jax.compute_dtype bfloat16 \
    --jax.prealloc True \
    --agent.dyn.rssm.gp.prior_lam "${PRIOR_LAM}" \
    --agent.dyn.rssm.gp.post_lam "${POST_LAM}" \
    --agent.dyn.rssm.gp.sample_frac "${SAMPLE_FRAC}" \
    --agent.dyn.rssm.gp.gp_rows "${GP_ROWS}" \
    --agent.dyn.rssm.gp.gp_rows_random "${GP_ROWS_RANDOM}" \
    ${DECAY_ARGS}

  echo "=== Completed seed ${seed} ==="
  echo ""
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "All seeds finished."
echo "Experiment: ${EXP_SUFFIX}"
echo "Prior lambda: ${PRIOR_LAM}"
echo "Posterior lambda: ${POST_LAM}"
echo "Decay type: ${DECAY_TYPE}"
echo "=========================================="
