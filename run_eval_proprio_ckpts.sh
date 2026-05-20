#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DreamerV3 Evaluation Script
# Evaluates local checkpoint directories produced by GPLD-MBRL training runs.
# ==============================================================================

# Configuration
CHECKPOINT_BASE="${CHECKPOINT_BASE:-./checkpoints/proprio}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./evals/proprio_eval}"
EVAL_ENVS="${EVAL_ENVS:-2}"           # Number of parallel eval environments
EVAL_STEPS="${EVAL_STEPS:-10020}"      # Total eval steps (~20 episodes with 501 steps each)
TARGET_CKPT_STEP="${TARGET_CKPT_STEP:-1000000}"  # Target checkpoint step

# Trial run options
TRIAL_RUN="${TRIAL_RUN:-false}"
TRIAL_GAME="${TRIAL_GAME:-dmc_acrobot_swingup}"
TRIAL_CONFIG="${TRIAL_CONFIG:-baseline}"  # "lipschitz" or "baseline"
TRIAL_SEED="${TRIAL_SEED:-0}"

# List mode - just show what would be evaluated without running
LIST_ONLY="${LIST_ONLY:-false}"

# ==============================================================================
# Functions
# ==============================================================================

# Extract game name from folder name (e.g., dmc_acrobot_swingup, dmc_acrobot_swingup_sparse)
# Handles sparse variants correctly by including _sparse in the game name
extract_game_name() {
    local folder_name="$1"

    # Split by underscore and find 'dmc' position
    # Then collect game parts until we hit a config keyword
    local result=""
    local in_game=false
    local IFS='_'
    read -ra parts <<< "$folder_name"

    for part in "${parts[@]}"; do
        # Start collecting after 'dmc'
        if [[ "$part" == "dmc" ]]; then
            in_game=true
            result="dmc"
            continue
        fi

        if [[ "$in_game" == true ]]; then
            # Stop if we hit a config keyword
            # Config keywords: lipschitz, baseline, prior*, post*, h[0-9]+, seed[0-9]+
            if [[ "$part" == "lipschitz" ]] || \
               [[ "$part" == "baseline" ]] || \
               [[ "$part" =~ ^prior[0-9] ]] || \
               [[ "$part" =~ ^post[0-9] ]] || \
               [[ "$part" =~ ^h[0-9]+ ]] || \
               [[ "$part" =~ ^seed[0-9]+ ]] || \
               [[ "$part" == "sqrt" ]] || \
               [[ "$part" == "tr512" ]]; then
                break
            fi

            # Add this part to the game name
            result="${result}_${part}"
        fi
    done

    echo "$result"
}

# Extract config type (lipschitz or baseline)
extract_config_type() {
    local folder_name="$1"
    if [[ "$folder_name" == *"lipschitz"* ]]; then
        echo "lipschitz"
    elif [[ "$folder_name" == *"baseline"* ]]; then
        echo "baseline"
    else
        echo "unknown"
    fi
}

# Extract seed from folder name
extract_seed() {
    local folder_name="$1"
    echo "$folder_name" | grep -oP 'seed\K[0-9]+'
}

# Find the best checkpoint close to target step
find_checkpoint() {
    local exp_dir="$1"
    local ckpt_dir="${exp_dir}/ckpt"

    # Priority order:
    # 1. Exact match: step_00250000_step/
    # 2. Final checkpoint: step_*_final/
    # 3. Closest available checkpoint

    if [[ -d "${ckpt_dir}/step_00${TARGET_CKPT_STEP}_step" ]]; then
        echo "${ckpt_dir}/step_00${TARGET_CKPT_STEP}_step"
        return 0
    fi

    # Look for final checkpoint (usually 255008)
    local final_ckpt
    final_ckpt=$(find "${ckpt_dir}" -maxdepth 1 -type d -name "step_*_final" 2>/dev/null | head -1)
    if [[ -n "$final_ckpt" ]]; then
        echo "$final_ckpt"
        return 0
    fi

    # Find closest checkpoint to target
    local closest_ckpt=""
    local min_diff=999999999

    for ckpt in "${ckpt_dir}"/step_*_step/; do
        if [[ -d "$ckpt" ]]; then
            # Extract step number
            local step_num
            step_num=$(basename "$ckpt" | grep -oP 'step_\K[0-9]+')
            if [[ -n "$step_num" ]]; then
                local diff=$((step_num > TARGET_CKPT_STEP ? step_num - TARGET_CKPT_STEP : TARGET_CKPT_STEP - step_num))
                if [[ $diff -lt $min_diff ]]; then
                    min_diff=$diff
                    closest_ckpt="$ckpt"
                fi
            fi
        fi
    done

    if [[ -n "$closest_ckpt" ]]; then
        echo "${closest_ckpt%/}"  # Remove trailing slash
        return 0
    fi

    return 1
}

# Run evaluation for a single experiment
run_eval() {
    local exp_dir="$1"
    local folder_name
    folder_name=$(basename "$exp_dir")

    local game_name
    game_name=$(extract_game_name "$folder_name")

    local config_type
    config_type=$(extract_config_type "$folder_name")

    local seed
    seed=$(extract_seed "$folder_name")

    # Find checkpoint (now from local storage)
    local ckpt_path
    if ! ckpt_path=$(find_checkpoint "$exp_dir"); then
        echo "WARNING: No checkpoint found for ${folder_name}, skipping"
        return 1
    fi

    local ckpt_step
    ckpt_step=$(basename "$ckpt_path" | grep -oP '[0-9]+' | head -1)

    echo ""
    echo "=========================================="
    echo "Evaluating: ${game_name}"
    echo "  Config: ${config_type}"
    echo "  Seed: ${seed}"
    echo "  Checkpoint: ${ckpt_path}"
    echo "  Checkpoint step: ${ckpt_step}"
    echo "=========================================="

    local output_dir="${EVAL_OUTPUT_DIR}/${config_type}/${game_name}/seed${seed}"
    mkdir -p "$output_dir"

    # Log start time
    local start_time
    start_time=$(date +%s)
    echo "Start time: $(date)"

    # Run evaluation from local checkpoint
    python -m dreamerv3.main \
        --logdir "${output_dir}" \
        --script eval_only \
        --configs dmc_proprio size12m \
        --task "${game_name}" \
        --agent.imag_length 25 \
        --run.envs "${EVAL_ENVS}" \
        --run.steps "${EVAL_STEPS}" \
        --run.from_checkpoint "${ckpt_path}" \
        --seed "${seed}" \
        --jax.platform cuda \
        --jax.compute_dtype bfloat16 \
        --jax.prealloc True

    local exit_code=$?

    # Log end time and duration
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo "End time: $(date)"
    echo "Duration: ${duration} seconds ($((duration / 60)) minutes)"
    echo "Results saved to: ${output_dir}"
    echo ""

    return $exit_code
}

# ==============================================================================
# Main Script
# ==============================================================================

echo "=============================================="
echo "DreamerV3 Checkpoint Evaluation Script"
echo "=============================================="
echo "Checkpoint directory: ${CHECKPOINT_BASE}"
echo "Eval output directory: ${EVAL_OUTPUT_DIR}"
echo "Eval environments: ${EVAL_ENVS}"
echo "Eval steps: ${EVAL_STEPS}"
echo "Target checkpoint step: ${TARGET_CKPT_STEP}"
echo ""
echo "Modes:"
echo "  Trial run: ${TRIAL_RUN}"
echo "  List only: ${LIST_ONLY}"
echo "=============================================="
echo ""

# ==============================================================================
# LIST MODE: Show all experiments without running
# ==============================================================================
if [[ "$LIST_ONLY" == "true" ]]; then
    echo ">>> LIST MODE - Showing all experiments <<<"
    echo ""

    # Collect unique games
    declare -A games_lipschitz=()
    declare -A games_baseline=()

    for folder in "${CHECKPOINT_BASE}"/*; do
        if [[ -d "$folder" ]]; then
            folder_name=$(basename "$folder")
            game=$(extract_game_name "$folder_name")
            config=$(extract_config_type "$folder_name")
            seed=$(extract_seed "$folder_name")

            if [[ "$config" == "lipschitz" ]]; then
                games_lipschitz["$game"]="${games_lipschitz[$game]:-}seed${seed} "
            elif [[ "$config" == "baseline" ]]; then
                games_baseline["$game"]="${games_baseline[$game]:-}seed${seed} "
            fi
        fi
    done

    echo "LIPSCHITZ experiments:"
    echo "----------------------"
    for game in $(echo "${!games_lipschitz[@]}" | tr ' ' '\n' | sort); do
        printf "  %-35s seeds: %s\n" "$game" "${games_lipschitz[$game]}"
    done

    echo ""
    echo "BASELINE experiments:"
    echo "---------------------"
    for game in $(echo "${!games_baseline[@]}" | tr ' ' '\n' | sort); do
        printf "  %-35s seeds: %s\n" "$game" "${games_baseline[$game]}"
    done

    echo ""
    echo "Total lipschitz games: ${#games_lipschitz[@]}"
    echo "Total baseline games: ${#games_baseline[@]}"
    exit 0
fi

# Create output directory
mkdir -p "${EVAL_OUTPUT_DIR}"

if [[ "$TRIAL_RUN" == "true" ]]; then
    # ==============================================================================
    # TRIAL RUN: Single checkpoint evaluation
    # ==============================================================================
    echo ">>> TRIAL RUN MODE <<<"
    echo "Game: ${TRIAL_GAME}"
    echo "Config: ${TRIAL_CONFIG}"
    echo "Seed: ${TRIAL_SEED}"
    echo ""

    # Find matching experiment folder
    exp_folder=""
    for folder in "${CHECKPOINT_BASE}"/*; do
        if [[ -d "$folder" ]]; then
            folder_name=$(basename "$folder")
            game=$(extract_game_name "$folder_name")
            config=$(extract_config_type "$folder_name")
            seed=$(extract_seed "$folder_name")

            if [[ "$game" == "$TRIAL_GAME" && "$config" == "$TRIAL_CONFIG" && "$seed" == "$TRIAL_SEED" ]]; then
                exp_folder="$folder"
                break
            fi
        fi
    done

    if [[ -z "$exp_folder" ]]; then
        echo "ERROR: Could not find experiment folder matching:"
        echo "  Game: ${TRIAL_GAME}"
        echo "  Config: ${TRIAL_CONFIG}"
        echo "  Seed: ${TRIAL_SEED}"
        echo ""
        echo "Available experiments:"
        for folder in "${CHECKPOINT_BASE}"/*"${TRIAL_GAME}"*"${TRIAL_CONFIG}"*; do
            if [[ -d "$folder" ]]; then
                echo "  $(basename "$folder")"
            fi
        done
        exit 1
    fi

    echo "Found experiment: $(basename "$exp_folder")"
    echo ""

    run_eval "$exp_folder"

    echo ""
    echo "=============================================="
    echo "TRIAL RUN COMPLETE"
    echo "=============================================="

else
    # ==============================================================================
    # FULL RUN: Evaluate all checkpoints
    # ==============================================================================

    # Count total experiments
    total_experiments=0
    for folder in "${CHECKPOINT_BASE}"/*; do
        if [[ -d "$folder" ]]; then
            ((total_experiments++)) || true
        fi
    done

    echo "Found ${total_experiments} experiments to evaluate"
    echo ""

    # Track progress
    current=0
    successful=0
    failed=0
    skipped=0

    # Process each experiment
    for folder in "${CHECKPOINT_BASE}"/*; do
        if [[ -d "$folder" ]]; then
            ((current++)) || true
            echo ""
            echo "=============================================="
            echo "Progress: ${current}/${total_experiments}"
            echo "=============================================="

            if run_eval "$folder"; then
                ((successful++)) || true
            else
                ((failed++)) || true
            fi
        fi
    done

    # Summary
    echo ""
    echo "=============================================="
    echo "EVALUATION COMPLETE"
    echo "=============================================="
    echo "Total experiments: ${total_experiments}"
    echo "Successful: ${successful}"
    echo "Failed: ${failed}"
    echo "Results saved to: ${EVAL_OUTPUT_DIR}"
    echo "=============================================="
fi
