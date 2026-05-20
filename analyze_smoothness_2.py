#!/usr/bin/env python3
"""
Analyze the smoothness of prior and posterior distributions in a trained DreamerV3 model.

This script:
1. Loads a trained model checkpoint (or multiple checkpoints)
2. Collects trajectory from DMC environment with full model states:
   - Physics states, observations, actions
   - Deterministic states h (computed via dynamics)
   - Stochastic states z (sampled from posterior)
   - Prior distribution logits p(z|h)
   - Posterior distribution logits q(z|h,e)
   - Encoder outputs e
3. Perturbs observations directly: o2 = o1 + epsilon * noise
4. Compares posterior distributions: q(z|h, e2) vs q(z|h, e1)
5. Also analyzes Lipschitz properties by perturbing h directly

Key analysis:
- For each sample with observation o1 and stored h:
- Perturb observation: o2 = o1 + epsilon * noise
- Encode both: e1 = enc(o1), e2 = enc(o2)
- Compute posteriors: q(z|h, e1) and q(z|h, e2)
- Compare distributions via KL divergence, TV distance, etc.

Note: Terminal states are skipped during collection. If an episode terminates
before enough samples are collected, a new episode is started.

Usage:
    # Analyze latest checkpoint only
    python copydreamerv3/analyze_smoothness.py -c <path_to_experiment_folder>

    # Analyze all checkpoints in the ckpt directory
    python copydreamerv3/analyze_smoothness.py -c <path_to_experiment_folder> --all_checkpoints

    # Analyze all checkpoints, but skip every 5 (analyze every 5th checkpoint)
    python copydreamerv3/analyze_smoothness.py -c <path_to_experiment_folder> --all_checkpoints --skip_steps 5

    # Analyze a specific checkpoint step
    python copydreamerv3/analyze_smoothness.py -c <path_to_experiment_folder> --checkpoint_step 100000

Output:
    - When using --all_checkpoints:
        - smoothness_analysis_all_ckpts/smoothness_step_XXXXXXXX.png  (per-checkpoint plots)
        - smoothness_analysis_all_ckpts/smoothness_step_XXXXXXXX.npz  (per-checkpoint data)
        - smoothness_training_progression.png  (metrics across training)
        - smoothness_all_checkpoints.npz  (combined data)
    - Single checkpoint:
        - smoothness_analysis.png
        - smoothness_results.npz
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import elements
import matplotlib.pyplot as plt

f32 = jnp.float32


def load_config(config_path):
    """Load config from yaml file."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return elements.Config(config)


def load_checkpoint(ckpt_dir):
    """Load model checkpoint."""
    ckpt_dir = Path(ckpt_dir)

    # Find latest checkpoint
    latest_file = ckpt_dir / 'latest'
    if latest_file.exists():
        with open(latest_file) as f:
            latest = f.read().strip()
        ckpt_path = ckpt_dir / latest
    else:
        # Find the most recent checkpoint folder
        subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
        if subdirs:
            def get_step(d):
                name = d.name
                if name.startswith('step_'):
                    try:
                        return int(name.split('_')[1])
                    except:
                        return 0
                return 0
            subdirs.sort(key=get_step, reverse=True)
            ckpt_path = subdirs[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    print(f"Loading checkpoint from: {ckpt_path}")

    # Load agent state
    agent_pkl = ckpt_path / 'agent.pkl'
    if agent_pkl.exists():
        with open(agent_pkl, 'rb') as f:
            agent_data = pickle.load(f)
        print(f"  Loaded agent state")
        if isinstance(agent_data, dict):
            if 'params' in agent_data:
                print(f"  Found {len(agent_data['params'])} parameter groups")
            if 'counters' in agent_data:
                print(f"  Counters: {agent_data['counters']}")
    else:
        agent_data = None
        print(f"  Warning: No agent.pkl found")

    # Load step
    step_pkl = ckpt_path / 'step.pkl'
    if step_pkl.exists():
        with open(step_pkl, 'rb') as f:
            step = pickle.load(f)
        print(f"  Training step: {step}")
    else:
        step = 0

    return agent_data, step, ckpt_path


def create_environment(task_name):
    """Create DMC environment.

    Args:
        task_name: Task in format 'domain_task' (e.g., 'quadruped_walk')
    """
    from dm_control import suite

    print(f"\nCreating environment: {task_name}")
    domain, task = task_name.split('_', 1)

    # Handle special domain names
    if domain == 'cup':
        domain = 'ball_in_cup'

    env = suite.load(domain, task)
    print(f"  Domain: {domain}, Task: {task}")
    print(f"  Physics state dim: {env.physics.get_state().shape}")
    print(f"  Action spec: {env.action_spec()}")
    return env


def get_observation_from_state(env, camera_id=0, size=(64, 64)):
    """Get observation dict from current physics state."""
    timestep = env._task.get_observation(env.physics)

    obs = {}
    # Get proprioceptive observations
    for key, value in timestep.items():
        if isinstance(value, np.ndarray):
            obs[key] = value.astype(np.float32)
        elif isinstance(value, (np.floating, np.integer)):
            obs[key] = np.array([value], dtype=np.float32)
        elif isinstance(value, (float, int)):
            obs[key] = np.array([value], dtype=np.float32)

    # Get image observation
    obs['image'] = env.physics.render(*size, camera_id=camera_id).astype(np.float32) / 255.0 - 0.5

    return obs


def collect_states_and_observations(env, num_samples=100, camera_id=0, size=(64, 64)):
    """Collect physics states and corresponding observations.

    Note: This function only collects raw environment data.
    Model states (h, z, prior, posterior, encoder output) are computed
    separately in collect_trajectory_with_model_states().

    Returns:
        states: list of physics states
        observations: list of observation dicts
        actions: list of actions taken (action that LED to current state)
        prev_actions: list of previous actions (for dynamics computation)
    """
    print(f"\nCollecting {num_samples} state-observation pairs...")

    states = []
    observations = []
    actions = []
    prev_actions = []

    action_spec = env.action_spec()

    # Run episodes until we have enough samples
    collected = 0
    episode_count = 0

    while collected < num_samples:
        timestep = env.reset()
        episode_count += 1
        step_count = 0
        prev_action = np.zeros(action_spec.shape, dtype=np.float32)  # Initial action is zeros

        while not timestep.last() and collected < num_samples:
            # Get current state and observation
            state = env.physics.get_state().copy()
            obs = get_observation_from_state(env, camera_id, size)

            # Random action for next step
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum,
                size=action_spec.shape
            ).astype(np.float32)

            states.append(state)
            observations.append(obs)
            actions.append(action)  # Action to be taken FROM this state
            prev_actions.append(prev_action.copy())  # Action that LED to this state
            collected += 1
            step_count += 1

            # Step environment
            timestep = env.step(action)
            prev_action = action.copy()

        print(f"  Episode {episode_count}: collected {step_count} samples (total: {collected})")

    print(f"  Total samples: {len(states)}")
    print(f"  State vector dimension: {states[0].shape}")
    print(f"  Action dimension: {actions[0].shape}")

    # Print example state vector
    print(f"\n  Example state vector (first sample):")
    print(f"    {states[0][:10]}... (showing first 10 of {len(states[0])} elements)")

    return states, observations, actions, prev_actions


def collect_trajectory_with_model_states(env, params, config, num_samples=100,
                                          camera_id=0, size=(64, 64), rng=None):
    """Collect trajectory data along with model states.

    For each non-terminal state, collect:
    - Physics state s
    - Observation o
    - Action a (action taken FROM this state)
    - Previous action a_prev (action that LED to this state)
    - Deterministic state h
    - Stochastic state z
    - Prior distribution logits p(z|h)
    - Posterior distribution logits q(z|h,e)
    - Encoder output e

    Note: Terminal states are skipped. If episode terminates before enough
    samples are collected, a new episode is started.

    Returns:
        trajectory: dict with all collected data
    """
    print(f"\nCollecting {num_samples} trajectory samples with model states...")
    print("  (Skipping terminal states)")

    # Get model dimensions
    rssm_config = config.agent.dyn.rssm
    deter_dim = rssm_config.deter
    stoch = rssm_config.stoch
    classes = rssm_config.classes
    hidden = rssm_config.hidden
    blocks = rssm_config.blocks

    print(f"  Model dimensions: deter={deter_dim}, stoch={stoch}, classes={classes}")

    # Build model functions
    encode_fn = build_encoder_fn(params, config)
    prior_fn = build_prior_fn(params, config)
    posterior_fn = build_posterior_fn(params, config)
    dynamics_fn = build_dynamics_fn(params, config)

    action_spec = env.action_spec()
    action_dim = action_spec.shape[0]

    # Storage
    trajectory = {
        'physics_states': [],
        'observations': [],
        'actions': [],
        'prev_actions': [],
        'h_states': [],  # Deterministic states
        'z_states': [],  # Stochastic states (one-hot sampled)
        'prior_logits': [],  # Prior distribution logits
        'post_logits': [],  # Posterior distribution logits
        'encoder_outputs': [],  # Encoder embeddings
    }

    collected = 0
    episode_count = 0

    if rng is None:
        rng = jax.random.PRNGKey(42)

    while collected < num_samples:
        timestep = env.reset()
        episode_count += 1
        step_count = 0
        skipped_terminal = 0

        # Initialize model states for new episode
        h = jnp.zeros((deter_dim,), dtype=jnp.float32)
        z = jnp.zeros((stoch, classes), dtype=jnp.float32)
        prev_action = np.zeros(action_spec.shape, dtype=np.float32)

        while collected < num_samples:
            # Check if episode ended
            if timestep.last():
                skipped_terminal += 1
                print(f"  Episode {episode_count}: collected {step_count} samples, skipped 1 terminal (total: {collected})")
                break

            # Get current physics state and observation
            physics_state = env.physics.get_state().copy()
            obs = get_observation_from_state(env, camera_id, size)

            # Encode observation
            obs_batch = {k: v[None, ...] for k, v in obs.items()}  # Add batch dim
            e = encode_fn(obs_batch)[0]  # Remove batch dim

            # Compute h using dynamics: h_t = dynamics(h_{t-1}, z_{t-1}, a_{t-1})
            # For first step of episode, h stays as zeros (initialized above)
            # For subsequent steps, h is updated
            if step_count > 0:
                h = dynamics_fn(h[None, ...], z[None, ...], prev_action[None, ...])[0]

            # Compute prior and posterior
            prior_logits = prior_fn(h[None, ...])[0]  # p(z|h)
            post_logits = posterior_fn(h[None, ...], e[None, ...])[0]  # q(z|h,e)

            # Sample z from posterior
            rng, key = jax.random.split(rng)
            z = sample_from_logits(post_logits, key)

            # Random action for next step
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum,
                size=action_spec.shape
            ).astype(np.float32)

            # Store data
            trajectory['physics_states'].append(physics_state)
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['prev_actions'].append(prev_action.copy())
            trajectory['h_states'].append(np.array(h))
            trajectory['z_states'].append(np.array(z))
            trajectory['prior_logits'].append(np.array(prior_logits))
            trajectory['post_logits'].append(np.array(post_logits))
            trajectory['encoder_outputs'].append(np.array(e))

            collected += 1
            step_count += 1

            # Step environment
            timestep = env.step(action)
            prev_action = action.copy()

        if not timestep.last():
            print(f"  Episode {episode_count}: collected {step_count} samples (total: {collected})")

    # Convert lists to arrays
    trajectory['physics_states'] = np.stack(trajectory['physics_states'], axis=0)
    trajectory['actions'] = np.stack(trajectory['actions'], axis=0)
    trajectory['prev_actions'] = np.stack(trajectory['prev_actions'], axis=0)
    trajectory['h_states'] = np.stack(trajectory['h_states'], axis=0)
    trajectory['z_states'] = np.stack(trajectory['z_states'], axis=0)
    trajectory['prior_logits'] = np.stack(trajectory['prior_logits'], axis=0)
    trajectory['post_logits'] = np.stack(trajectory['post_logits'], axis=0)
    trajectory['encoder_outputs'] = np.stack(trajectory['encoder_outputs'], axis=0)
    # observations remain as list of dicts

    print(f"\n  Collection complete!")
    print(f"  Total samples: {collected}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Physics state shape: {trajectory['physics_states'].shape}")
    print(f"  h_states shape: {trajectory['h_states'].shape}")
    print(f"  z_states shape: {trajectory['z_states'].shape}")
    print(f"  prior_logits shape: {trajectory['prior_logits'].shape}")
    print(f"  post_logits shape: {trajectory['post_logits'].shape}")
    print(f"  encoder_outputs shape: {trajectory['encoder_outputs'].shape}")

    return trajectory


def sample_from_logits(logits, rng_key, unimix=0.01):
    """Sample one-hot vector from categorical distribution defined by logits.

    Args:
        logits: shape (stoch, classes)
        rng_key: JAX random key
        unimix: uniform mixture coefficient

    Returns:
        one_hot: shape (stoch, classes) - one-hot sampled vectors
    """
    # Add uniform mixture
    probs = jax.nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1 - unimix) * probs + unimix * uniform

    # Sample from each categorical
    stoch, classes = logits.shape
    keys = jax.random.split(rng_key, stoch)

    def sample_one(key, prob):
        idx = jax.random.categorical(key, jnp.log(prob + 1e-8))
        return jax.nn.one_hot(idx, classes)

    one_hot = jax.vmap(sample_one)(keys, probs)
    return one_hot


def perturb_state_and_get_observation(env, state, epsilon, camera_id=0, size=(64, 64), rng_key=None):
    """Perturb physics state by epsilon and get new observation.

    Args:
        env: DMC environment
        state: Original physics state
        epsilon: Perturbation magnitude
        camera_id: Camera for rendering
        size: Image size
        rng_key: JAX random key for noise

    Returns:
        perturbed_obs: Observation from perturbed state
        actual_epsilon: Actual L2 norm of perturbation applied
    """
    # Generate noise
    if rng_key is not None:
        noise = np.array(jax.random.normal(rng_key, state.shape))
    else:
        noise = np.random.randn(*state.shape)

    # Normalize noise and scale by epsilon
    noise = noise / (np.linalg.norm(noise) + 1e-8) * epsilon * np.sqrt(len(state))

    # Perturb state
    perturbed_state = state + noise
    actual_epsilon = np.linalg.norm(perturbed_state - state)

    # Set perturbed state in environment
    env.physics.set_state(perturbed_state)

    # Forward dynamics to make physics consistent (optional but recommended)
    try:
        env.physics.forward()
    except:
        pass  # Some states might be invalid

    # Get observation from perturbed state
    perturbed_obs = get_observation_from_state(env, camera_id, size)

    # Restore original state
    env.physics.set_state(state)
    try:
        env.physics.forward()
    except:
        pass

    return perturbed_obs, actual_epsilon


def preprocess_observations(obs_list):
    """Stack observations into batched arrays."""
    processed = {}

    for key in obs_list[0].keys():
        values = np.stack([obs[key] for obs in obs_list], axis=0)
        processed[key] = values

    return processed


def logits_to_probs(logits, unimix=0.01):
    """Convert logits to probabilities with uniform mixture.

    This implements the same distribution as embodied.jax.outs.Categorical:
    probs = (1 - unimix) * softmax(logits) + unimix * uniform

    Args:
        logits: shape (..., stoch, classes)
        unimix: uniform mixture coefficient (default 0.01)

    Returns:
        probs: shape (..., stoch, classes) - proper probability distributions
    """
    probs = jax.nn.softmax(logits, axis=-1)
    if unimix > 0:
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - unimix) * probs + unimix * uniform
    return probs


def apply_unimix_to_logits(logits, unimix=0.01):
    """Apply unimix and convert back to logits, matching Categorical behavior.

    This is what embodied.jax.outs.Categorical does in __init__:
    - Convert logits to probs via softmax
    - Mix with uniform distribution
    - Convert back to logits via log

    The resulting logits can then be used with log_softmax and softmax
    for numerically stable KL computation.
    """
    if unimix > 0:
        probs = jax.nn.softmax(logits, axis=-1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - unimix) * probs + unimix * uniform
        return jnp.log(probs)
    return logits


def kl_divergence_per_row(logits1, logits2, unimix=0.01):
    """Compute KL divergence for each row (stoch dimension) separately.

    This matches the implementation in embodied.jax.outs.Categorical.kl():
    - First apply unimix to convert logits
    - Then use log_softmax for numerically stable log-probabilities
    - KL(P||Q) = sum_x P(x) * (log P(x) - log Q(x))

    Args:
        logits1: shape (batch, stoch, classes) - "P" distribution
        logits2: shape (batch, stoch, classes) - "Q" distribution
        unimix: uniform mixture coefficient

    Returns:
        kl_per_row: shape (batch, stoch) - KL divergence for each of the 32 rows (always >= 0)
    """
    # Apply unimix transformation (like Categorical.__init__)
    logits1 = apply_unimix_to_logits(logits1, unimix)
    logits2 = apply_unimix_to_logits(logits2, unimix)

    # Use log_softmax for numerical stability (like Categorical.kl)
    logprob1 = jax.nn.log_softmax(logits1, axis=-1)
    logprob2 = jax.nn.log_softmax(logits2, axis=-1)
    prob1 = jax.nn.softmax(logits1, axis=-1)

    # KL for each row: sum over classes dimension
    kl_per_row = jnp.sum(prob1 * (logprob1 - logprob2), axis=-1)

    # Clamp to >= 0 (KL is theoretically non-negative, but floating-point errors
    # can produce tiny negative values when distributions are nearly identical)
    kl_per_row = jnp.maximum(kl_per_row, 0.0)
    return kl_per_row


def kl_divergence_categorical(logits1, logits2, unimix=0.01):
    """Compute total KL divergence (sum over all 32 rows).

    Args:
        logits1: shape (batch, stoch, classes) - "P" distribution
        logits2: shape (batch, stoch, classes) - "Q" distribution
        unimix: uniform mixture coefficient

    Returns:
        kl_total: shape (batch,) - total KL (sum over stoch dimension)
    """
    kl_per_row = kl_divergence_per_row(logits1, logits2, unimix)
    return jnp.sum(kl_per_row, axis=-1)


def symmetric_kl_per_row(logits1, logits2, unimix=0.01):
    """Compute symmetric KL divergence for each row separately."""
    kl_forward = kl_divergence_per_row(logits1, logits2, unimix)
    kl_backward = kl_divergence_per_row(logits2, logits1, unimix)
    return 0.5 * (kl_forward + kl_backward)


def symmetric_kl(logits1, logits2, unimix=0.01):
    """Compute symmetric KL divergence (sum over all rows)."""
    return 0.5 * (kl_divergence_categorical(logits1, logits2, unimix) +
                  kl_divergence_categorical(logits2, logits1, unimix))


def total_variation_per_row(logits1, logits2, unimix=0.01):
    """Compute total variation distance for each row separately."""
    probs1 = logits_to_probs(logits1, unimix)
    probs2 = logits_to_probs(logits2, unimix)
    tv_per_row = 0.5 * jnp.sum(jnp.abs(probs1 - probs2), axis=-1)
    return tv_per_row


def total_variation_distance(logits1, logits2, unimix=0.01):
    """Compute total variation distance (sum over all rows)."""
    tv_per_row = total_variation_per_row(logits1, logits2, unimix)
    return jnp.sum(tv_per_row, axis=-1)


def hellinger_per_row(logits1, logits2, unimix=0.01):
    """Compute Hellinger distance for each row separately."""
    probs1 = logits_to_probs(logits1, unimix)
    probs2 = logits_to_probs(logits2, unimix)
    h2_per_row = 0.5 * jnp.sum((jnp.sqrt(probs1) - jnp.sqrt(probs2))**2, axis=-1)
    return jnp.sqrt(h2_per_row)


def hellinger_distance(logits1, logits2, unimix=0.01):
    """Compute Hellinger distance (sqrt of sum of squared per-row H^2)."""
    probs1 = logits_to_probs(logits1, unimix)
    probs2 = logits_to_probs(logits2, unimix)
    h2 = 0.5 * jnp.sum((jnp.sqrt(probs1) - jnp.sqrt(probs2))**2, axis=-1)
    h = jnp.sqrt(jnp.sum(h2, axis=-1))
    return h


def build_encoder_fn(params, config):
    """Build a function to encode observations using loaded parameters."""
    enc_config = config.agent.enc.simple
    layers = enc_config.layers
    units = enc_config.units

    def encode(obs_dict):
        """Encode observations to get tokens."""
        # Concatenate all observation features (excluding image for now - proprioceptive only)
        features = []
        for key, value in sorted(obs_dict.items()):
            if key in ['is_first', 'is_last', 'is_terminal', 'reward', 'image']:
                continue
            # Ensure 2D: (batch, features)
            if len(value.shape) == 1:
                # Scalar feature: (batch,) -> (batch, 1)
                value = value[:, None]
            elif len(value.shape) > 2:
                # Higher dimensional: flatten to (batch, -1)
                value = value.reshape(value.shape[0], -1)
            features.append(value)

        if not features:
            # If no proprioceptive features, use flattened image
            if 'image' in obs_dict:
                features = [obs_dict['image'].reshape(obs_dict['image'].shape[0], -1)]

        x = jnp.concatenate(features, axis=-1)

        # Apply encoder layers
        for i in range(layers):
            kernel = params.get(f'enc/mlp{i}/kernel', None)
            bias = params.get(f'enc/mlp{i}/bias', None)
            if kernel is not None:
                x = x @ kernel
                if bias is not None:
                    x = x + bias

            # RMS normalization
            scale = params.get(f'enc/mlp{i}norm/scale', None)
            if scale is not None:
                ms = jnp.mean(x**2, axis=-1, keepdims=True)
                x = x * jax.lax.rsqrt(ms + 1e-8) * scale

            x = jax.nn.silu(x)

        return x

    return encode


def build_prior_fn(params, config):
    """Build a function to compute prior logits from deterministic state."""
    rssm_config = config.agent.dyn.rssm
    imglayers = rssm_config.imglayers
    stoch = rssm_config.stoch
    classes = rssm_config.classes

    def prior(h):
        """Compute prior logits from deterministic state h."""
        x = h
        for i in range(imglayers):
            kernel = params.get(f'dyn/prior{i}/kernel', None)
            bias = params.get(f'dyn/prior{i}/bias', None)
            if kernel is not None:
                x = x @ kernel
                if bias is not None:
                    x = x + bias
            scale = params.get(f'dyn/prior{i}norm/scale', None)
            if scale is not None:
                ms = jnp.mean(x**2, axis=-1, keepdims=True)
                x = x * jax.lax.rsqrt(ms + 1e-8) * scale
            x = jax.nn.silu(x)

        kernel = params.get('dyn/priorlogit/kernel', None)
        bias = params.get('dyn/priorlogit/bias', None)
        if kernel is not None:
            x = x @ kernel
            if bias is not None:
                x = x + bias

        logits = x.reshape((*x.shape[:-1], stoch, classes))
        return logits

    return prior


def build_posterior_fn(params, config):
    """Build a function to compute posterior logits from h and encoder output."""
    rssm_config = config.agent.dyn.rssm
    obslayers = rssm_config.obslayers
    stoch = rssm_config.stoch
    classes = rssm_config.classes
    absolute = rssm_config.absolute

    def posterior(h, e):
        """Compute posterior logits from h (deter) and e (encoder output)."""
        if absolute:
            x = e
        else:
            x = jnp.concatenate([h, e], axis=-1)

        for i in range(obslayers):
            kernel = params.get(f'dyn/obs{i}/kernel', None)
            bias = params.get(f'dyn/obs{i}/bias', None)
            if kernel is not None:
                x = x @ kernel
                if bias is not None:
                    x = x + bias
            scale = params.get(f'dyn/obs{i}norm/scale', None)
            if scale is not None:
                ms = jnp.mean(x**2, axis=-1, keepdims=True)
                x = x * jax.lax.rsqrt(ms + 1e-8) * scale
            x = jax.nn.silu(x)

        kernel = params.get('dyn/obslogit/kernel', None)
        bias = params.get('dyn/obslogit/bias', None)
        if kernel is not None:
            x = x @ kernel
            if bias is not None:
                x = x + bias

        logits = x.reshape((*x.shape[:-1], stoch, classes))
        return logits

    return posterior


def build_dynamics_fn(params, config):
    """Build a function to compute next deterministic state h from (h, z, action).

    This implements the GRU-like dynamics core of the RSSM:
    h_t = dynamics(h_{t-1}, z_{t-1}, a_{t-1})

    Note: dynhid and dyngru layers use BlockLinear, which has a block-diagonal
    kernel structure with shape (blocks, in_per_block, out_per_block).
    """
    import einops

    rssm_config = config.agent.dyn.rssm
    deter = rssm_config.deter
    hidden = rssm_config.hidden
    blocks = rssm_config.blocks
    dynlayers = rssm_config.dynlayers

    def block_linear(x, kernel, bias=None):
        """Apply BlockLinear: kernel has shape (blocks, in_per_block, out_per_block)."""
        g = kernel.shape[0]
        in_per_block = kernel.shape[1]
        out_total = g * kernel.shape[2]
        # Reshape x: (batch, g * in_per_block) -> (batch, g, in_per_block)
        x = x.reshape((*x.shape[:-1], g, in_per_block))
        # Block-diagonal matmul: einsum '...ki,kio->...ko'
        x = jnp.einsum('...ki,kio->...ko', x, kernel)
        # Reshape back: (batch, g, out_per_block) -> (batch, out_total)
        x = x.reshape((*x.shape[:-2], out_total))
        if bias is not None:
            x = x + bias
        return x

    def dynamics(h, z, action):
        """Compute next h from current h, z, and action.

        Args:
            h: shape (batch, deter) - current deterministic state
            z: shape (batch, stoch, classes) - current stochastic state (one-hot)
            action: shape (batch, action_dim) - action taken

        Returns:
            h_next: shape (batch, deter) - next deterministic state
        """
        # Flatten z: (batch, stoch, classes) -> (batch, stoch * classes)
        z_flat = z.reshape((z.shape[0], -1))

        # Normalize action
        action = action / jnp.maximum(1.0, jnp.abs(action))

        g = blocks
        flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
        group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

        # Input projections (regular Linear layers)
        # dynin0: h -> hidden
        kernel = params.get('dyn/dynin0/kernel', None)
        bias = params.get('dyn/dynin0/bias', None)
        if kernel is not None:
            x0 = h @ kernel
            if bias is not None:
                x0 = x0 + bias
        else:
            x0 = h
        scale = params.get('dyn/dynin0norm/scale', None)
        if scale is not None:
            ms = jnp.mean(x0**2, axis=-1, keepdims=True)
            x0 = x0 * jax.lax.rsqrt(ms + 1e-8) * scale
        x0 = jax.nn.silu(x0)

        # dynin1: z -> hidden
        kernel = params.get('dyn/dynin1/kernel', None)
        bias = params.get('dyn/dynin1/bias', None)
        if kernel is not None:
            x1 = z_flat @ kernel
            if bias is not None:
                x1 = x1 + bias
        else:
            x1 = z_flat
        scale = params.get('dyn/dynin1norm/scale', None)
        if scale is not None:
            ms = jnp.mean(x1**2, axis=-1, keepdims=True)
            x1 = x1 * jax.lax.rsqrt(ms + 1e-8) * scale
        x1 = jax.nn.silu(x1)

        # dynin2: action -> hidden
        kernel = params.get('dyn/dynin2/kernel', None)
        bias = params.get('dyn/dynin2/bias', None)
        if kernel is not None:
            x2 = action @ kernel
            if bias is not None:
                x2 = x2 + bias
        else:
            x2 = action
        scale = params.get('dyn/dynin2norm/scale', None)
        if scale is not None:
            ms = jnp.mean(x2**2, axis=-1, keepdims=True)
            x2 = x2 * jax.lax.rsqrt(ms + 1e-8) * scale
        x2 = jax.nn.silu(x2)

        # Concatenate and replicate for block structure
        x = jnp.concatenate([x0, x1, x2], axis=-1)  # (batch, 3*hidden)
        x = x[..., None, :].repeat(g, axis=-2)  # (batch, g, 3*hidden)
        x = group2flat(jnp.concatenate([flat2group(h), x], axis=-1))  # (batch, deter + 3*hidden*g)

        # Hidden layers (BlockLinear)
        for i in range(dynlayers):
            kernel = params.get(f'dyn/dynhid{i}/kernel', None)
            bias = params.get(f'dyn/dynhid{i}/bias', None)
            if kernel is not None:
                x = block_linear(x, kernel, bias)
            scale = params.get(f'dyn/dynhid{i}norm/scale', None)
            if scale is not None:
                ms = jnp.mean(x**2, axis=-1, keepdims=True)
                x = x * jax.lax.rsqrt(ms + 1e-8) * scale
            x = jax.nn.silu(x)

        # GRU gates (BlockLinear)
        kernel = params.get('dyn/dyngru/kernel', None)
        bias = params.get('dyn/dyngru/bias', None)
        if kernel is not None:
            x = block_linear(x, kernel, bias)

        # Split into reset, candidate, update gates
        gates = jnp.split(flat2group(x), 3, axis=-1)
        reset_gate, cand, update = [group2flat(gate) for gate in gates]
        reset_gate = jax.nn.sigmoid(reset_gate)
        cand = jnp.tanh(reset_gate * cand)
        update = jax.nn.sigmoid(update - 1)  # Bias toward remembering

        # GRU update
        h_next = update * cand + (1 - update) * h

        return h_next

    return dynamics


def analyze_physics_state_perturbation(env, params, config, trajectory, epsilon_fractions, rng,
                                        camera_id=0, size=(64, 64), unimix=0.01):
    """
    Analyze smoothness by perturbing physics states and tracking changes through the full pipeline.

    Pipeline: s1 -> o1 -> e1 -> (with h1, z1, a1) -> h2 via dynamics

    For each sample:
    1. Original: s1 -> o1 -> e1, with stored h1, z1, a1
    2. Perturbed: s2 = s1 + δ, o2 = o1 + δ (same perturbation added to observation)
    3. Encode perturbed observation: e2 = enc(o2)
    4. Compare encoder outputs: ||e2 - e1||_2
    5. Compare posterior distributions: q(z|h, e2) vs q(z|h, e1)

    Note: h_next is NOT changed in this analysis because we use z_base (the z sampled
    during trajectory collection) for both baseline and perturbed cases. This ensures
    a fair comparison without sampling randomness. The h_next perturbation analysis
    is done in analyze_direct_perturbation() where we explicitly perturb h and a.

    Epsilon is a fraction of the mean norm:
    - ||δ||_2 = ε × mean(||s||_2)

    This allows comparing perturbation effects relative to the scale of the data.
    """
    print("\n" + "="*80)
    print("ANALYZING PHYSICS STATE PERTURBATION SMOOTHNESS")
    print("="*80)
    print(f"  Using unimix={unimix} for categorical distribution")
    print(f"\n  Pipeline: s -> o -> e -> (h, z, a) -> h_next via dynamics")
    print(f"  Perturbation: o2 = o1 + δ (same δ as added to physics state)")
    print(f"  Prior p(z|h) depends on: h")
    print(f"  Posterior q(z|h,e) depends on: h and e")

    # Build model functions
    print("\nBuilding model functions...")
    encode_fn = build_encoder_fn(params, config)
    prior_fn = build_prior_fn(params, config)
    posterior_fn = build_posterior_fn(params, config)
    dynamics_fn = build_dynamics_fn(params, config)

    # Get dimensions
    rssm_config = config.agent.dyn.rssm
    deter_dim = rssm_config.deter
    stoch = rssm_config.stoch
    classes = rssm_config.classes

    # Extract data from trajectory
    observations = trajectory['observations']
    h_states = jnp.array(trajectory['h_states'])
    z_states = jnp.array(trajectory['z_states'])
    prior_logits_stored = jnp.array(trajectory['prior_logits'])
    post_logits_stored = jnp.array(trajectory['post_logits'])
    encoder_outputs = jnp.array(trajectory['encoder_outputs'])
    actions = jnp.array(trajectory['actions'])
    prev_actions = jnp.array(trajectory['prev_actions'])
    physics_states = trajectory['physics_states']

    batch_size = len(observations)
    state_dim = physics_states[0].shape[0]
    action_dim = actions.shape[-1]
    encoder_dim = encoder_outputs.shape[-1]

    # Compute mean norms for scaling
    state_norms = np.array([np.linalg.norm(s) for s in physics_states])
    mean_state_norm = np.mean(state_norms)
    mean_h_norm = float(jnp.mean(jnp.linalg.norm(h_states, axis=-1)))
    mean_e_norm = float(jnp.mean(jnp.linalg.norm(encoder_outputs, axis=-1)))

    print(f"\n  Dimensions:")
    print(f"    Batch size: {batch_size}")
    print(f"    Physics state dim: {state_dim}")
    print(f"    Action dim: {action_dim}")
    print(f"    Deterministic state (h) dim: {deter_dim}")
    print(f"    Encoder output (e) dim: {encoder_dim}")
    print(f"    Stochastic state (z): {stoch} x {classes} = {stoch * classes}")

    print(f"\n  Mean norms (for epsilon scaling):")
    print(f"    mean(||s||_2) = {mean_state_norm:.4f}")
    print(f"    mean(||h||_2) = {mean_h_norm:.4f}")
    print(f"    mean(||e||_2) = {mean_e_norm:.4f}")

    # Print example data
    print(f"\n  Example data (first sample):")
    print(f"    Physics state s (first 10): {physics_states[0][:10]}")
    print(f"    ||s||_2 = {np.linalg.norm(physics_states[0]):.4f}")
    print(f"    h (first 10): {h_states[0, :10]}")
    print(f"    e (first 10): {encoder_outputs[0, :10]}")
    print(f"    action a: {actions[0]}")

    # Compute baseline distributions
    prior_logits_base = prior_fn(h_states)
    post_logits_base = posterior_fn(h_states, encoder_outputs)

    # Baseline h_next
    h_next_base = dynamics_fn(h_states, z_states, actions)
    mean_h_next_norm = float(jnp.mean(jnp.linalg.norm(h_next_base, axis=-1)))
    print(f"    mean(||h_next||_2) = {mean_h_next_norm:.4f}")

    # Results storage
    results = {
        'epsilon_fraction': [],  # Fraction of mean norm
        'epsilon_absolute': [],  # Actual L2 norm of perturbation
        'stoch': stoch,
        'classes': classes,
        'unimix': unimix,
        'state_dim': state_dim,
        'deter_dim': deter_dim,
        'encoder_dim': encoder_dim,
        'action_dim': action_dim,
        'mean_state_norm': mean_state_norm,
        'mean_h_norm': mean_h_norm,
        'mean_e_norm': mean_e_norm,
        # State perturbation tracking
        'state_perturbation_vectors': [],
        'state_perturbation_l2': [],
        # Observation and encoder changes
        'encoder_l2_mean': [], 'encoder_l2_std': [],
        # h_next perturbation (caused by z perturbation through dynamics)
        'h_next_perturbation_l2_mean': [], 'h_next_perturbation_l2_std': [],
        'h_next_perturbation_pct_mean': [], 'h_next_perturbation_pct_std': [],
        # Prior metrics (depends on h only)
        'prior_kl_mean': [], 'prior_kl_std': [],
        'prior_tv_mean': [], 'prior_tv_std': [],
        # Posterior metrics (depends on h and e)
        'post_kl_mean': [], 'post_kl_std': [],
        'post_tv_mean': [], 'post_tv_std': [],
    }

    # Analyze for each epsilon fraction
    for eps_idx, eps_frac in enumerate(epsilon_fractions):
        # Compute actual epsilon based on mean state norm
        eps_absolute = eps_frac * mean_state_norm

        print(f"\n  {'='*60}")
        print(f"  Testing epsilon = {eps_frac:.4e} × mean(||s||) = {eps_absolute:.4e}")
        print(f"  {'='*60}")

        # Generate perturbation noise for all samples
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, batch_size)

        # Store perturbation info
        perturbation_vectors = []
        state_perturbation_l2 = []
        perturbed_observations = []

        for i in range(batch_size):
            state = physics_states[i]
            orig_obs = observations[i]

            # Generate normalized noise for state
            noise = np.array(jax.random.normal(keys[i], state.shape))
            noise_normalized = noise / (np.linalg.norm(noise) + 1e-8)

            # Scale by epsilon_absolute (fraction of mean state norm)
            perturbation = noise_normalized * eps_absolute

            perturbation_vectors.append(perturbation)
            state_perturbation_l2.append(np.linalg.norm(perturbation))

            # Create perturbed observation: o2 = o1 + δ (same δ as added to s)
            # We apply the perturbation to the proprioceptive features
            pert_obs = {}
            pert_idx = 0
            for obs_key, obs_val in orig_obs.items():
                if obs_key in ['is_first', 'is_last', 'is_terminal', 'reward']:
                    pert_obs[obs_key] = obs_val
                elif obs_key == 'image':
                    # Don't perturb image directly - would need physics rendering
                    pert_obs[obs_key] = obs_val
                else:
                    # Add scaled perturbation to proprioceptive features
                    obs_size = obs_val.size
                    if pert_idx + obs_size <= len(perturbation):
                        obs_pert = perturbation[pert_idx:pert_idx + obs_size].reshape(obs_val.shape)
                        pert_obs[obs_key] = obs_val + obs_pert.astype(obs_val.dtype)
                        pert_idx += obs_size
                    else:
                        # Wrap around or use remaining perturbation
                        pert_obs[obs_key] = obs_val

            perturbed_observations.append(pert_obs)

        # Print example perturbation vector for first epsilon
        if eps_idx == 0:
            print(f"\n    === PERTURBATION VECTOR EXAMPLES (first sample) ===")
            print(f"    Original state s1 (first 15):")
            print(f"      {physics_states[0][:15]}")
            print(f"    Perturbation vector δ (first 15):")
            print(f"      {perturbation_vectors[0][:15]}")
            print(f"    Perturbed state s2 = s1 + δ (first 15):")
            print(f"      {physics_states[0][:15] + perturbation_vectors[0][:15]}")
            print(f"    ||δ||_2 = {state_perturbation_l2[0]:.6f}")
            print(f"    ||δ||_2 / ||s1||_2 = {state_perturbation_l2[0] / (np.linalg.norm(physics_states[0]) + 1e-8):.6f} ({100*state_perturbation_l2[0] / (np.linalg.norm(physics_states[0]) + 1e-8):.4f}%)")

        # Encode perturbed observations
        pert_obs_batch = preprocess_observations(perturbed_observations)
        e_perturbed = encode_fn(pert_obs_batch)

        # Compute encoder perturbation
        encoder_l2 = jnp.sqrt(jnp.sum((encoder_outputs - e_perturbed)**2, axis=-1))

        # Compute perturbed posterior
        post_logits_pert = posterior_fn(h_states, e_perturbed)

        # Use the SAME z as the original trajectory (z_states) for fair comparison
        # This isolates the effect of observation perturbation on encoder output,
        # without additional randomness from sampling different z values.
        # Since z_base = z_perturbed and h, a are unchanged:
        #   h_next_base = dynamics(h, z_base, a)
        #   h_next_pert = dynamics(h, z_base, a) = h_next_base
        # Therefore, there is no h_next perturbation in this analysis.
        # The h_next perturbation is analyzed in the direct perturbation section
        # where we explicitly perturb h and a.

        # h_next is unchanged when using z_base (no sampling randomness)
        h_next_l2 = jnp.zeros(batch_size)
        h_next_norms = jnp.linalg.norm(h_next_base, axis=-1)
        h_next_pct = jnp.zeros(batch_size)

        # Compute distribution metrics
        post_kl = kl_divergence_categorical(post_logits_base, post_logits_pert, unimix)
        post_tv = total_variation_distance(post_logits_base, post_logits_pert, unimix)

        # Prior is unchanged (h is fixed)
        prior_kl = jnp.zeros(batch_size)
        prior_tv = jnp.zeros(batch_size)

        # Store results
        results['epsilon_fraction'].append(float(eps_frac))
        results['epsilon_absolute'].append(float(eps_absolute))
        results['state_perturbation_vectors'].append(np.stack(perturbation_vectors, axis=0))
        results['state_perturbation_l2'].append(np.array(state_perturbation_l2))

        results['encoder_l2_mean'].append(float(jnp.mean(encoder_l2)))
        results['encoder_l2_std'].append(float(jnp.std(encoder_l2)))

        results['h_next_perturbation_l2_mean'].append(float(jnp.mean(h_next_l2)))
        results['h_next_perturbation_l2_std'].append(float(jnp.std(h_next_l2)))
        results['h_next_perturbation_pct_mean'].append(float(jnp.mean(h_next_pct)))
        results['h_next_perturbation_pct_std'].append(float(jnp.std(h_next_pct)))

        results['prior_kl_mean'].append(float(jnp.mean(prior_kl)))
        results['prior_kl_std'].append(float(jnp.std(prior_kl)))
        results['prior_tv_mean'].append(float(jnp.mean(prior_tv)))
        results['prior_tv_std'].append(float(jnp.std(prior_tv)))

        results['post_kl_mean'].append(float(jnp.mean(post_kl)))
        results['post_kl_std'].append(float(jnp.std(post_kl)))
        results['post_tv_mean'].append(float(jnp.mean(post_tv)))
        results['post_tv_std'].append(float(jnp.std(post_tv)))

        # Print summary
        print(f"\n    State perturbation ||δ||_2: {np.mean(state_perturbation_l2):.4e}")
        print(f"    Encoder change ||e2-e1||_2: {float(jnp.mean(encoder_l2)):.4e}")
        print(f"    h_next change: 0 (using z_base from trajectory, not sampling)")
        print(f"    Posterior KL: {float(jnp.mean(post_kl)):.4e}")
        print(f"    Posterior TV: {float(jnp.mean(post_tv)):.4e}")

    # Store trajectory data
    results['h_states'] = np.array(h_states)
    results['z_states'] = np.array(z_states)
    results['encoder_outputs'] = np.array(encoder_outputs)
    results['actions'] = np.array(actions)
    results['prev_actions'] = np.array(prev_actions)
    results['physics_states'] = physics_states
    results['prior_logits_stored'] = np.array(prior_logits_stored)
    results['post_logits_stored'] = np.array(post_logits_stored)

    return results


def analyze_direct_perturbation(params, config, trajectory, epsilon_fractions, rng, unimix=0.01):
    """
    Analyze smoothness by directly perturbing model inputs.

    This tests the local Lipschitz property of prior and posterior networks:

    1. PRIOR p(z|h):
       - Perturb h only (since prior only depends on h)
       - Perturb (h, a) jointly to see effect on dynamics output

    2. POSTERIOR q(z|h,e):
       - Perturb h only (with e fixed)
       - Perturb e only (with h fixed)
       - Perturb (h, e) jointly

    Epsilon is a fraction of the mean norm:
    - ||δh||_2 = ε × mean(||h||_2)
    - ||δe||_2 = ε × mean(||e||_2)
    - ||δa||_2 = ε × mean(||a||_2)

    Uses the stored h, e, a values from the trajectory.
    Per-row KL and TV statistics are computed for each of the stoch independent categoricals.
    """
    print("\n" + "="*80)
    print("ANALYZING DIRECT INPUT PERTURBATION (LIPSCHITZ ANALYSIS)")
    print("="*80)
    print(f"  Using unimix={unimix} for categorical distribution")
    print(f"  Epsilon = fraction of mean norm (e.g., 0.01 = 1% of mean norm)")

    # Build model functions
    prior_fn = build_prior_fn(params, config)
    posterior_fn = build_posterior_fn(params, config)
    dynamics_fn = build_dynamics_fn(params, config)

    rssm_config = config.agent.dyn.rssm
    deter_dim = rssm_config.deter
    stoch = rssm_config.stoch
    classes = rssm_config.classes

    # Use stored values from trajectory
    h = jnp.array(trajectory['h_states'])
    e = jnp.array(trajectory['encoder_outputs'])
    z = jnp.array(trajectory['z_states'])
    a = jnp.array(trajectory['actions'])
    batch_size = h.shape[0]
    encoder_dim = e.shape[-1]
    action_dim = a.shape[-1]

    # Compute mean norms for scaling
    mean_h_norm = float(jnp.mean(jnp.linalg.norm(h, axis=-1)))
    mean_e_norm = float(jnp.mean(jnp.linalg.norm(e, axis=-1)))
    mean_a_norm = float(jnp.mean(jnp.linalg.norm(a, axis=-1)))

    print(f"\n  Dimensions:")
    print(f"    Batch size: {batch_size}")
    print(f"    h (deter) dim: {deter_dim}")
    print(f"    e (encoder) dim: {encoder_dim}")
    print(f"    a (action) dim: {action_dim}")
    print(f"    z (stoch): {stoch} x {classes}")

    print(f"\n  Mean norms (for epsilon scaling):")
    print(f"    mean(||h||_2) = {mean_h_norm:.4f}")
    print(f"    mean(||e||_2) = {mean_e_norm:.4f}")
    print(f"    mean(||a||_2) = {mean_a_norm:.4f}")

    # Results storage
    results = {
        'epsilon_fraction': [],  # Fraction of mean norm
        'epsilon_h_absolute': [],  # Actual ||δh||_2
        'epsilon_e_absolute': [],  # Actual ||δe||_2
        'epsilon_a_absolute': [],  # Actual ||δa||_2
        'stoch': stoch,
        'classes': classes,
        'unimix': unimix,
        'deter_dim': deter_dim,
        'encoder_dim': encoder_dim,
        'action_dim': action_dim,
        'mean_h_norm': mean_h_norm,
        'mean_e_norm': mean_e_norm,
        'mean_a_norm': mean_a_norm,

        # === PRIOR ANALYSIS: p(z|h) ===
        # Perturb h only
        'prior_h_kl_mean': [], 'prior_h_kl_std': [],
        'prior_h_tv_mean': [], 'prior_h_tv_std': [],
        'prior_h_kl_per_row_all': [],
        'prior_h_perturbation_vectors': [],

        # === POSTERIOR ANALYSIS: q(z|h,e) ===
        # Perturb h only (e fixed)
        'post_h_kl_mean': [], 'post_h_kl_std': [],
        'post_h_tv_mean': [], 'post_h_tv_std': [],
        'post_h_kl_per_row_all': [],

        # Perturb e only (h fixed)
        'post_e_kl_mean': [], 'post_e_kl_std': [],
        'post_e_tv_mean': [], 'post_e_tv_std': [],
        'post_e_kl_per_row_all': [],
        'post_e_perturbation_vectors': [],

        # Perturb (h, e) jointly
        'post_he_kl_mean': [], 'post_he_kl_std': [],
        'post_he_tv_mean': [], 'post_he_tv_std': [],
        'post_he_kl_per_row_all': [],

        # === DYNAMICS ANALYSIS: h_next = dynamics(h, z, a) ===
        # Perturb (h, a) and see h_next change
        'dynamics_h_perturbation_l2_mean': [], 'dynamics_h_perturbation_l2_std': [],
        'dynamics_h_perturbation_pct_mean': [], 'dynamics_h_perturbation_pct_std': [],
    }

    # Baseline distributions
    prior_logits_base = prior_fn(h)
    post_logits_base = posterior_fn(h, e)

    # Baseline dynamics output
    h_next_base = dynamics_fn(h, z, a)
    mean_h_next_norm = float(jnp.mean(jnp.linalg.norm(h_next_base, axis=-1)))

    print(f"    mean(||h_next||_2) = {mean_h_next_norm:.4f}")

    for eps_idx, eps_frac in enumerate(epsilon_fractions):
        # Compute actual epsilon for each input type
        eps_h = eps_frac * mean_h_norm
        eps_e = eps_frac * mean_e_norm
        eps_a = eps_frac * mean_a_norm

        print(f"\n  {'='*60}")
        print(f"  Testing epsilon = {eps_frac:.4e} (= {100*eps_frac:.4f}% of mean norm)")
        print(f"    ||δh||_2 = {eps_h:.4e}")
        print(f"    ||δe||_2 = {eps_e:.4e}")
        print(f"    ||δa||_2 = {eps_a:.4e}")
        print(f"  {'='*60}")

        rng, key1, key2, key3, key4 = jax.random.split(rng, 5)

        # === Generate perturbation noise ===

        # h perturbation: ||δh||_2 = eps_h
        noise_h = jax.random.normal(key1, h.shape)
        noise_h_norms = jnp.linalg.norm(noise_h, axis=-1, keepdims=True) + 1e-8
        delta_h = noise_h / noise_h_norms * eps_h
        h_pert = h + delta_h

        # e perturbation: ||δe||_2 = eps_e
        noise_e = jax.random.normal(key2, e.shape)
        noise_e_norms = jnp.linalg.norm(noise_e, axis=-1, keepdims=True) + 1e-8
        delta_e = noise_e / noise_e_norms * eps_e
        e_pert = e + delta_e

        # a perturbation: ||δa||_2 = eps_a
        noise_a = jax.random.normal(key3, a.shape)
        noise_a_norms = jnp.linalg.norm(noise_a, axis=-1, keepdims=True) + 1e-8
        delta_a = noise_a / noise_a_norms * eps_a
        a_pert = a + delta_a

        # Print example perturbation vectors for first epsilon
        if eps_idx == 0:
            print(f"\n    === PERTURBATION VECTORS (first sample) ===")
            print(f"    δh (first 10): {delta_h[0, :10]}")
            print(f"    ||δh||_2 = {float(jnp.linalg.norm(delta_h[0])):.6f} ({100*float(jnp.linalg.norm(delta_h[0]))/(float(jnp.linalg.norm(h[0]))+1e-8):.4f}% of ||h||)")
            print(f"    δe (first 10): {delta_e[0, :10]}")
            print(f"    ||δe||_2 = {float(jnp.linalg.norm(delta_e[0])):.6f} ({100*float(jnp.linalg.norm(delta_e[0]))/(float(jnp.linalg.norm(e[0]))+1e-8):.4f}% of ||e||)")
            print(f"    δa: {delta_a[0]}")
            print(f"    ||δa||_2 = {float(jnp.linalg.norm(delta_a[0])):.6f} ({100*float(jnp.linalg.norm(delta_a[0]))/(float(jnp.linalg.norm(a[0]))+1e-8):.4f}% of ||a||)")

        # === PRIOR: Perturb h only ===
        prior_logits_h_pert = prior_fn(h_pert)
        prior_h_kl = kl_divergence_categorical(prior_logits_base, prior_logits_h_pert, unimix)
        prior_h_tv = total_variation_distance(prior_logits_base, prior_logits_h_pert, unimix)
        prior_h_kl_per_row = kl_divergence_per_row(prior_logits_base, prior_logits_h_pert, unimix)

        # === POSTERIOR: Perturb h only (e fixed) ===
        post_logits_h_pert = posterior_fn(h_pert, e)
        post_h_kl = kl_divergence_categorical(post_logits_base, post_logits_h_pert, unimix)
        post_h_tv = total_variation_distance(post_logits_base, post_logits_h_pert, unimix)
        post_h_kl_per_row = kl_divergence_per_row(post_logits_base, post_logits_h_pert, unimix)

        # === POSTERIOR: Perturb e only (h fixed) ===
        post_logits_e_pert = posterior_fn(h, e_pert)
        post_e_kl = kl_divergence_categorical(post_logits_base, post_logits_e_pert, unimix)
        post_e_tv = total_variation_distance(post_logits_base, post_logits_e_pert, unimix)
        post_e_kl_per_row = kl_divergence_per_row(post_logits_base, post_logits_e_pert, unimix)

        # === POSTERIOR: Perturb (h, e) jointly ===
        post_logits_he_pert = posterior_fn(h_pert, e_pert)
        post_he_kl = kl_divergence_categorical(post_logits_base, post_logits_he_pert, unimix)
        post_he_tv = total_variation_distance(post_logits_base, post_logits_he_pert, unimix)
        post_he_kl_per_row = kl_divergence_per_row(post_logits_base, post_logits_he_pert, unimix)

        # === DYNAMICS: Perturb (h, a) and see h_next change ===
        h_next_pert = dynamics_fn(h_pert, z, a_pert)
        dynamics_h_pert_l2 = jnp.sqrt(jnp.sum((h_next_base - h_next_pert)**2, axis=-1))
        h_next_norms = jnp.linalg.norm(h_next_base, axis=-1)
        dynamics_h_pert_pct = 100 * dynamics_h_pert_l2 / (h_next_norms + 1e-8)

        # Store results
        results['epsilon_fraction'].append(float(eps_frac))
        results['epsilon_h_absolute'].append(float(eps_h))
        results['epsilon_e_absolute'].append(float(eps_e))
        results['epsilon_a_absolute'].append(float(eps_a))

        # Prior (h perturbed)
        results['prior_h_kl_mean'].append(float(jnp.mean(prior_h_kl)))
        results['prior_h_kl_std'].append(float(jnp.std(prior_h_kl)))
        results['prior_h_tv_mean'].append(float(jnp.mean(prior_h_tv)))
        results['prior_h_tv_std'].append(float(jnp.std(prior_h_tv)))
        results['prior_h_kl_per_row_all'].append(np.array(prior_h_kl_per_row))
        results['prior_h_perturbation_vectors'].append(np.array(delta_h))

        # Posterior (h perturbed only)
        results['post_h_kl_mean'].append(float(jnp.mean(post_h_kl)))
        results['post_h_kl_std'].append(float(jnp.std(post_h_kl)))
        results['post_h_tv_mean'].append(float(jnp.mean(post_h_tv)))
        results['post_h_tv_std'].append(float(jnp.std(post_h_tv)))
        results['post_h_kl_per_row_all'].append(np.array(post_h_kl_per_row))

        # Posterior (e perturbed only)
        results['post_e_kl_mean'].append(float(jnp.mean(post_e_kl)))
        results['post_e_kl_std'].append(float(jnp.std(post_e_kl)))
        results['post_e_tv_mean'].append(float(jnp.mean(post_e_tv)))
        results['post_e_tv_std'].append(float(jnp.std(post_e_tv)))
        results['post_e_kl_per_row_all'].append(np.array(post_e_kl_per_row))
        results['post_e_perturbation_vectors'].append(np.array(delta_e))

        # Posterior (h,e perturbed jointly)
        results['post_he_kl_mean'].append(float(jnp.mean(post_he_kl)))
        results['post_he_kl_std'].append(float(jnp.std(post_he_kl)))
        results['post_he_tv_mean'].append(float(jnp.mean(post_he_tv)))
        results['post_he_tv_std'].append(float(jnp.std(post_he_tv)))
        results['post_he_kl_per_row_all'].append(np.array(post_he_kl_per_row))

        # Dynamics
        results['dynamics_h_perturbation_l2_mean'].append(float(jnp.mean(dynamics_h_pert_l2)))
        results['dynamics_h_perturbation_l2_std'].append(float(jnp.std(dynamics_h_pert_l2)))
        results['dynamics_h_perturbation_pct_mean'].append(float(jnp.mean(dynamics_h_pert_pct)))
        results['dynamics_h_perturbation_pct_std'].append(float(jnp.std(dynamics_h_pert_pct)))

        # Print summary
        print(f"\n    PRIOR p(z|h) with h perturbed:")
        print(f"      KL: {float(jnp.mean(prior_h_kl)):.4e} ± {float(jnp.std(prior_h_kl)):.4e}")
        print(f"      TV: {float(jnp.mean(prior_h_tv)):.4e} ± {float(jnp.std(prior_h_tv)):.4e}")

        print(f"\n    POSTERIOR q(z|h,e) - h perturbed only:")
        print(f"      KL: {float(jnp.mean(post_h_kl)):.4e} ± {float(jnp.std(post_h_kl)):.4e}")
        print(f"      TV: {float(jnp.mean(post_h_tv)):.4e} ± {float(jnp.std(post_h_tv)):.4e}")

        print(f"\n    POSTERIOR q(z|h,e) - e perturbed only:")
        print(f"      KL: {float(jnp.mean(post_e_kl)):.4e} ± {float(jnp.std(post_e_kl)):.4e}")
        print(f"      TV: {float(jnp.mean(post_e_tv)):.4e} ± {float(jnp.std(post_e_tv)):.4e}")

        print(f"\n    POSTERIOR q(z|h,e) - (h,e) perturbed jointly:")
        print(f"      KL: {float(jnp.mean(post_he_kl)):.4e} ± {float(jnp.std(post_he_kl)):.4e}")
        print(f"      TV: {float(jnp.mean(post_he_tv)):.4e} ± {float(jnp.std(post_he_tv)):.4e}")

        print(f"\n    DYNAMICS h_next = f(h,z,a) with (h,a) perturbed:")
        print(f"      ||Δh_next||_2: {float(jnp.mean(dynamics_h_pert_l2)):.4e} ± {float(jnp.std(dynamics_h_pert_l2)):.4e}")
        print(f"      % change: {float(jnp.mean(dynamics_h_pert_pct)):.4f}% ± {float(jnp.std(dynamics_h_pert_pct)):.4f}%")

    # Convert per-row data to arrays
    results['prior_h_kl_per_row_all'] = np.stack(results['prior_h_kl_per_row_all'], axis=0)
    results['post_h_kl_per_row_all'] = np.stack(results['post_h_kl_per_row_all'], axis=0)
    results['post_e_kl_per_row_all'] = np.stack(results['post_e_kl_per_row_all'], axis=0)
    results['post_he_kl_per_row_all'] = np.stack(results['post_he_kl_per_row_all'], axis=0)

    # Store trajectory data
    results['h_states'] = np.array(h)
    results['encoder_outputs'] = np.array(e)
    results['z_states'] = np.array(z)
    results['actions'] = np.array(a)

    return results


def plot_results(state_results, direct_results, output_path):
    """Generate comprehensive plots for physics state and direct perturbation analysis."""

    # Figure 1: Physics State Perturbation Analysis (2x2 grid)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Physics State Perturbation Analysis\no2 = o1 + δ (same δ as added to physics state)\nUsing z_base from trajectory (no sampling)', fontsize=14)

    eps = state_results['epsilon_fraction']
    stoch = state_results.get('stoch', 32)

    # [0,0] State perturbation magnitude
    ax = axes1[0, 0]
    state_l2_means = [np.mean(l2) for l2 in state_results['state_perturbation_l2']]
    state_l2_stds = [np.std(l2) for l2 in state_results['state_perturbation_l2']]
    ax.errorbar(eps, state_l2_means, yerr=state_l2_stds,
                fmt='o-', capsize=3, label='||δ||₂', color='blue')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('||δ||₂')
    ax.set_title('State Perturbation Magnitude')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # [0,1] Encoder change from state perturbation
    ax = axes1[0, 1]
    ax.errorbar(eps, state_results['encoder_l2_mean'], yerr=state_results['encoder_l2_std'],
                fmt='o-', capsize=3, label='||e2 - e1||₂', color='green')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('Encoder L2 Distance')
    ax.set_title('Encoder Output Change')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # [1,0] Posterior KL (e perturbed, h fixed)
    ax = axes1[1, 0]
    ax.errorbar(eps, state_results['post_kl_mean'], yerr=state_results['post_kl_std'],
                fmt='o-', capsize=3, label='Posterior KL', color='orange')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Posterior KL: q(z|h,e2) vs q(z|h,e1)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # [1,1] Posterior TV (e perturbed, h fixed)
    ax = axes1[1, 1]
    ax.errorbar(eps, state_results['post_tv_mean'], yerr=state_results['post_tv_std'],
                fmt='o-', capsize=3, label='Posterior TV', color='red')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('Total Variation Distance')
    ax.set_title('Posterior TV: q(z|h,e2) vs q(z|h,e1)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved state perturbation plot to: {output_path}")

    # Figure 2: Direct Perturbation Analysis (3x2 grid)
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 15))
    fig2.suptitle('Direct Input Perturbation Analysis (Lipschitz)\nε = fraction of mean norm', fontsize=14)

    eps_d = direct_results['epsilon_fraction']

    # Row 0: Prior Analysis p(z|h)
    ax = axes2[0, 0]
    ax.errorbar(eps_d, direct_results['prior_h_kl_mean'], yerr=direct_results['prior_h_kl_std'],
                fmt='o-', capsize=3, label='KL', color='purple')
    ax.set_xlabel('ε (fraction of mean ||h||)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Prior p(z|h): h perturbed')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes2[0, 1]
    ax.errorbar(eps_d, direct_results['prior_h_tv_mean'], yerr=direct_results['prior_h_tv_std'],
                fmt='o-', capsize=3, label='TV', color='purple')
    ax.set_xlabel('ε (fraction of mean ||h||)')
    ax.set_ylabel('Total Variation')
    ax.set_title('Prior p(z|h): h perturbed (TV)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Row 1: Posterior Analysis - separate perturbations
    ax = axes2[1, 0]
    ax.errorbar(eps_d, direct_results['post_h_kl_mean'], yerr=direct_results['post_h_kl_std'],
                fmt='o-', capsize=3, label='h perturbed only', color='crimson')
    ax.errorbar(eps_d, direct_results['post_e_kl_mean'], yerr=direct_results['post_e_kl_std'],
                fmt='s--', capsize=3, label='e perturbed only', color='darkorange')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Posterior q(z|h,e): Separate Perturbations')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes2[1, 1]
    ax.errorbar(eps_d, direct_results['post_h_tv_mean'], yerr=direct_results['post_h_tv_std'],
                fmt='o-', capsize=3, label='h perturbed only', color='crimson')
    ax.errorbar(eps_d, direct_results['post_e_tv_mean'], yerr=direct_results['post_e_tv_std'],
                fmt='s--', capsize=3, label='e perturbed only', color='darkorange')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('Total Variation')
    ax.set_title('Posterior q(z|h,e): Separate Perturbations (TV)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Row 2: Joint perturbation and dynamics
    ax = axes2[2, 0]
    ax.errorbar(eps_d, direct_results['post_he_kl_mean'], yerr=direct_results['post_he_kl_std'],
                fmt='o-', capsize=3, label='(h,e) joint', color='darkgreen')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Posterior q(z|h,e): Joint (h,e) Perturbation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes2[2, 1]
    ax.errorbar(eps_d, direct_results['dynamics_h_perturbation_pct_mean'],
                yerr=direct_results['dynamics_h_perturbation_pct_std'],
                fmt='o-', capsize=3, label='% change in ||h_next||', color='navy')
    ax.set_xlabel('ε (applied to h and a)')
    ax.set_ylabel('% Change in ||h_next||')
    ax.set_title('Dynamics h_next = f(h,z,a): Effect of (h,a) Perturbation')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    direct_plot_path = str(output_path).replace('.png', '_direct.png')
    plt.savefig(direct_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved direct perturbation plot to: {direct_plot_path}")

    # Figure 3: Comparison plot
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('KL and TV Comparison: Prior vs Posterior\nε = fraction of mean norm', fontsize=14)

    ax = axes3[0]
    ax.errorbar(eps_d, direct_results['prior_h_kl_mean'], yerr=direct_results['prior_h_kl_std'],
                fmt='o-', capsize=3, label='Prior (h pert)', color='purple')
    ax.errorbar(eps_d, direct_results['post_h_kl_mean'], yerr=direct_results['post_h_kl_std'],
                fmt='s-', capsize=3, label='Post (h pert)', color='crimson')
    ax.errorbar(eps_d, direct_results['post_e_kl_mean'], yerr=direct_results['post_e_kl_std'],
                fmt='^-', capsize=3, label='Post (e pert)', color='darkorange')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes3[1]
    ax.errorbar(eps_d, direct_results['prior_h_tv_mean'], yerr=direct_results['prior_h_tv_std'],
                fmt='o-', capsize=3, label='Prior (h pert)', color='purple')
    ax.errorbar(eps_d, direct_results['post_h_tv_mean'], yerr=direct_results['post_h_tv_std'],
                fmt='s-', capsize=3, label='Post (h pert)', color='crimson')
    ax.errorbar(eps_d, direct_results['post_e_tv_mean'], yerr=direct_results['post_e_tv_std'],
                fmt='^-', capsize=3, label='Post (e pert)', color='darkorange')
    ax.set_xlabel('ε (fraction of mean norm)')
    ax.set_ylabel('Total Variation')
    ax.set_title('Total Variation Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    comparison_plot_path = str(output_path).replace('.png', '_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {comparison_plot_path}")


def print_summary(state_results, direct_results):
    """Print summary tables for physics state and direct perturbation analysis."""
    print("\n" + "="*120)
    print("ANALYSIS SUMMARY")
    print("="*120)

    stoch = state_results.get('stoch', 32)
    classes = state_results.get('classes', 64)
    unimix = state_results.get('unimix', 0.01)

    print(f"\n  Distribution: {stoch} independent categoricals, each with {classes} classes")
    print(f"  Uniform mixture (unimix): {unimix}")
    print(f"  Epsilon = fraction of mean norm")

    # Print mean norms
    mean_s_norm = state_results.get('mean_state_norm', 0)
    mean_h_norm = direct_results.get('mean_h_norm', 0)
    mean_e_norm = direct_results.get('mean_e_norm', 0)
    print(f"\n  Mean norms: ||s||={mean_s_norm:.4f}, ||h||={mean_h_norm:.4f}, ||e||={mean_e_norm:.4f}")

    # ============ PHYSICS STATE PERTURBATION ============
    print("\n" + "="*120)
    print("PHYSICS STATE PERTURBATION EFFECTS")
    print("  Pipeline: s1 → o1 → e1   (original)")
    print("            s2 = s1 + δ, o2 = o1 + δ → e2   (perturbed)")
    print("  Posterior q(z|h,e) changes due to e change (h is fixed)")
    print("  Note: h_next is unchanged (using z_base from trajectory, not sampling from perturbed posterior)")
    print("="*120)

    print("\n  TABLE 1: Physics State Perturbation Summary")
    print("  " + "-"*100)
    print(f"  {'ε (frac)':<12} {'ε×||s||':<12} {'||e2-e1||₂':<14} {'Post KL':<14} {'Post TV':<14}")
    print("  " + "-"*100)
    for i, eps_frac in enumerate(state_results['epsilon_fraction']):
        eps_abs = state_results['epsilon_absolute'][i]
        print(f"  {eps_frac:<12.4e} "
              f"{eps_abs:<12.4e} "
              f"{state_results['encoder_l2_mean'][i]:<14.4e} "
              f"{state_results['post_kl_mean'][i]:<14.4e} "
              f"{state_results['post_tv_mean'][i]:<14.4e}")

    # Print example perturbation vector
    if len(state_results['state_perturbation_vectors']) > 0:
        delta = state_results['state_perturbation_vectors'][0][0]  # First epsilon, first sample
        eps_frac = state_results['epsilon_fraction'][0]
        eps_abs = state_results['epsilon_absolute'][0]
        print(f"\n  Example perturbation vector δ (first sample, ε={eps_frac:.4e} = {100*eps_frac:.4f}%):")
        print(f"    δ (first 20 elements): {delta[:20]}")
        print(f"    ||δ||₂ = {np.linalg.norm(delta):.6f} (target: {eps_abs:.6f})")

    # ============ DIRECT PERTURBATION ============
    print("\n" + "="*120)
    print("DIRECT INPUT PERTURBATION (LIPSCHITZ ANALYSIS)")
    print("  Prior p(z|h): depends only on h")
    print("  Posterior q(z|h,e): depends on h and e")
    print("  Dynamics h_next = f(h,z,a): depends on h, z, a")
    print("  Epsilon = fraction of mean norm: ||δh||₂ = ε × mean(||h||₂)")
    print("="*120)

    deter_dim = direct_results.get('deter_dim', 1024)
    encoder_dim = direct_results.get('encoder_dim', 1024)
    action_dim = direct_results.get('action_dim', 6)

    print(f"\n  Dimensions: h={deter_dim}, e={encoder_dim}, a={action_dim}")

    print("\n  TABLE 2: Prior p(z|h) - h Perturbation")
    print("  " + "-"*90)
    print(f"  {'ε (frac)':<12} {'||δh||₂':<12} {'KL':<14} {'TV':<14}")
    print("  " + "-"*90)
    for i, eps_frac in enumerate(direct_results['epsilon_fraction']):
        eps_h = direct_results['epsilon_h_absolute'][i]
        print(f"  {eps_frac:<12.4e} "
              f"{eps_h:<12.4e} "
              f"{direct_results['prior_h_kl_mean'][i]:<14.4e} "
              f"{direct_results['prior_h_tv_mean'][i]:<14.4e}")

    print("\n  TABLE 3: Posterior q(z|h,e) - Separate Perturbations")
    print("  " + "-"*100)
    print(f"  {'ε (frac)':<12} {'KL (h pert)':<14} {'TV (h pert)':<14} {'KL (e pert)':<14} {'TV (e pert)':<14}")
    print("  " + "-"*100)
    for i, eps_frac in enumerate(direct_results['epsilon_fraction']):
        print(f"  {eps_frac:<12.4e} "
              f"{direct_results['post_h_kl_mean'][i]:<14.4e} "
              f"{direct_results['post_h_tv_mean'][i]:<14.4e} "
              f"{direct_results['post_e_kl_mean'][i]:<14.4e} "
              f"{direct_results['post_e_tv_mean'][i]:<14.4e}")

    print("\n  TABLE 4: Posterior q(z|h,e) - Joint (h,e) Perturbation")
    print("  " + "-"*60)
    print(f"  {'ε (frac)':<12} {'KL':<14} {'TV':<14}")
    print("  " + "-"*60)
    for i, eps_frac in enumerate(direct_results['epsilon_fraction']):
        print(f"  {eps_frac:<12.4e} "
              f"{direct_results['post_he_kl_mean'][i]:<14.4e} "
              f"{direct_results['post_he_tv_mean'][i]:<14.4e}")

    print("\n  TABLE 5: Dynamics h_next = f(h,z,a) - (h,a) Perturbation")
    print("  " + "-"*70)
    print(f"  {'ε (frac)':<12} {'||Δh_next||₂':<14} {'% change':<14}")
    print("  " + "-"*70)
    for i, eps_frac in enumerate(direct_results['epsilon_fraction']):
        print(f"  {eps_frac:<12.4e} "
              f"{direct_results['dynamics_h_perturbation_l2_mean'][i]:<14.4e} "
              f"{direct_results['dynamics_h_perturbation_pct_mean'][i]:<14.4f}")

    # Print example perturbation vectors
    if len(direct_results['prior_h_perturbation_vectors']) > 0:
        delta_h = direct_results['prior_h_perturbation_vectors'][0][0]
        delta_e = direct_results['post_e_perturbation_vectors'][0][0]
        eps_frac = direct_results['epsilon_fraction'][0]
        print(f"\n  Example perturbation vectors (first sample, ε={eps_frac:.4e} = {100*eps_frac:.4f}%):")
        print(f"    δh (first 10): {delta_h[:10]}")
        print(f"    ||δh||₂ = {np.linalg.norm(delta_h):.6f}")
        print(f"    δe (first 10): {delta_e[:10]}")
        print(f"    ||δe||₂ = {np.linalg.norm(delta_e):.6f}")

    # ============ STORED DATA INFO ============
    print("\n" + "-"*120)
    print("STORED TRAJECTORY DATA")
    print("-"*120)
    print(f"  Physics states shape: {state_results['physics_states'].shape}")
    print(f"  h (deterministic) shape: {state_results['h_states'].shape}")
    print(f"  z (stochastic) shape: {state_results['z_states'].shape}")
    print(f"  e (encoder) shape: {state_results['encoder_outputs'].shape}")
    print(f"  actions shape: {state_results['actions'].shape}")

    h = state_results['h_states']
    print(f"\n  h statistics: mean={np.mean(h):.4f}, std={np.std(h):.4f}, ||h||₂={np.mean(np.linalg.norm(h, axis=-1)):.4f}")


def parse_task_from_folder(folder_name, config):
    """Parse task name from experiment folder name or config."""
    # Try to extract from folder name: TIMESTAMP_dmc_domain_task_...
    parts = folder_name.split('_')

    for i, part in enumerate(parts):
        if part == 'dmc' and i + 2 < len(parts):
            # Found dmc, next parts are domain and task
            domain = parts[i + 1]
            # Task might have underscores, so we need to be careful
            # Look for known suffixes like 'lipschitz', 'baseline', 'seed'
            task_parts = []
            for j in range(i + 2, len(parts)):
                if parts[j] in ['lipschitz', 'baseline', 'seed0', 'seed1', 'seed2',
                               'seed3', 'seed4', 'seed5', 'seed6', 'seed7', 'seed8', 'seed9']:
                    break
                if parts[j].startswith('seed'):
                    break
                if parts[j].startswith('prior'):
                    break
                if parts[j].startswith('post'):
                    break
                if parts[j] == 'h':
                    break
                task_parts.append(parts[j])

            if task_parts:
                task = '_'.join(task_parts)
                return f"{domain}_{task}"

    # Fall back to config
    if hasattr(config, 'task'):
        task_full = config.task
        if task_full.startswith('dmc_'):
            return task_full[4:]  # Remove 'dmc_' prefix
        return task_full

    return None


def find_all_checkpoints(ckpt_dir):
    """Find all checkpoint directories in the ckpt folder.

    Returns:
        List of tuples (step_number, checkpoint_path) sorted by step number
    """
    ckpt_dir = Path(ckpt_dir)
    checkpoints = []

    for item in ckpt_dir.iterdir():
        if item.is_dir():
            name = item.name
            # Parse step number from directory name
            # Format: step_XXXXXXXX_step or step_XXXXXXXX_final or timestamp format
            if name.startswith('step_'):
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        step = int(parts[1])
                        checkpoints.append((step, item))
                    except ValueError:
                        pass
            # Also check for timestamp format like 20260106T222449F306005
            elif len(name) > 15 and 'T' in name:
                # This is likely the initial checkpoint, treat as step 0
                checkpoints.append((0, item))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_checkpoint_from_path(ckpt_path):
    """Load model checkpoint from a specific path.

    Args:
        ckpt_path: Path to the checkpoint directory

    Returns:
        Tuple of (agent_data, step, ckpt_path)
    """
    ckpt_path = Path(ckpt_path)

    # Extract step number from path name
    step = 0
    name = ckpt_path.name
    if name.startswith('step_'):
        parts = name.split('_')
        if len(parts) >= 2:
            try:
                step = int(parts[1])
            except ValueError:
                pass

    # Load state from checkpoint directory
    agent_file = ckpt_path / 'agent.pkl'
    if agent_file.exists():
        with open(agent_file, 'rb') as f:
            agent_data = pickle.load(f)
        return agent_data, step, ckpt_path

    # Try loading from .pkl files in the directory
    pkl_files = list(ckpt_path.glob('*.pkl'))
    if pkl_files:
        with open(pkl_files[0], 'rb') as f:
            data = pickle.load(f)
        return data, step, ckpt_path

    return None, step, ckpt_path


def analyze_single_checkpoint(ckpt_path, config, task_name, env, epsilon_fractions, args, rng):
    """Analyze a single checkpoint.

    Args:
        ckpt_path: Path to the checkpoint directory
        config: Configuration object
        task_name: Task name string
        env: Environment instance
        epsilon_fractions: Array of epsilon fractions (of mean norm) to test
        args: Command line arguments
        rng: JAX random key

    Returns:
        Dictionary with results or None if failed
    """
    # Load checkpoint
    agent_data, step, _ = load_checkpoint_from_path(ckpt_path)

    # Extract parameters
    params = None
    if agent_data is not None and isinstance(agent_data, dict):
        params = agent_data.get('params', None)

    if params is None:
        print(f"  WARNING: No parameters found in checkpoint at step {step}")
        return None

    # Collect trajectory with model states
    rng, key = jax.random.split(rng)
    trajectory = collect_trajectory_with_model_states(
        env, params, config,
        num_samples=args.num_samples,
        camera_id=args.camera_id,
        size=(args.image_size, args.image_size),
        rng=key)

    # Run physics state perturbation analysis
    rng, key = jax.random.split(rng)
    state_results = analyze_physics_state_perturbation(
        env, params, config, trajectory, epsilon_fractions, key,
        camera_id=args.camera_id, size=(args.image_size, args.image_size),
        unimix=args.unimix)

    # Run direct perturbation analysis
    rng, key = jax.random.split(rng)
    direct_results = analyze_direct_perturbation(
        params, config, trajectory, epsilon_fractions, key, unimix=args.unimix)

    return {
        'step': step,
        'state_results': state_results,
        'direct_results': direct_results,
        'trajectory': trajectory,
        'rng': rng
    }


def plot_training_progression(all_results, epsilon_values, output_path, task_name):
    """Plot how smoothness metrics change across training.

    Args:
        all_results: List of results dictionaries from each checkpoint
        epsilon_values: Array of epsilon values used
        output_path: Path to save the plot
        task_name: Task name for plot title
    """
    # Extract data
    steps = [r['step'] for r in all_results]

    # Use the largest epsilon for the summary (most visible differences)
    eps_idx = -1  # Last epsilon (largest)
    eps_val = epsilon_values[eps_idx]

    # Extract metrics for each checkpoint
    obs_kl_means = []
    obs_kl_medians = []
    h_prior_kl_means = []
    h_post_kl_means = []

    for r in all_results:
        obs = r['obs_results']
        h = r['h_results']

        # Get per-row mean KL for largest epsilon
        obs_kl_means.append(obs['per_row_kl_stats'][eps_idx]['mean_of_means'])
        obs_kl_medians.append(obs['per_row_kl_stats'][eps_idx]['median_of_means'])
        h_prior_kl_means.append(h['prior_kl'][eps_idx])
        h_post_kl_means.append(h['per_row_kl_stats'][eps_idx]['mean_of_means'])

    # Convert to arrays
    steps = np.array(steps)
    obs_kl_means = np.array(obs_kl_means)
    obs_kl_medians = np.array(obs_kl_medians)
    h_prior_kl_means = np.array(h_prior_kl_means)
    h_post_kl_means = np.array(h_post_kl_means)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Smoothness Analysis Across Training - {task_name}\n(ε = {eps_val:.2e})', fontsize=14)

    # Plot 1: Observation perturbation KL (mean)
    ax = axes[0, 0]
    ax.plot(steps / 1000, obs_kl_means, 'b-o', markersize=4, label='Mean of row means')
    ax.plot(steps / 1000, obs_kl_medians, 'g--s', markersize=4, label='Median of row means')
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Per-Row KL Divergence')
    ax.set_title('Observation Perturbation: Posterior KL')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 2: H perturbation - Prior KL
    ax = axes[0, 1]
    ax.plot(steps / 1000, h_prior_kl_means, 'r-o', markersize=4)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Total KL Divergence')
    ax.set_title('H Perturbation: Prior KL (total)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: H perturbation - Posterior KL
    ax = axes[1, 0]
    ax.plot(steps / 1000, h_post_kl_means, 'm-o', markersize=4)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Per-Row KL Divergence')
    ax.set_title('H Perturbation: Posterior KL (per-row mean)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison
    ax = axes[1, 1]
    ax.plot(steps / 1000, obs_kl_means, 'b-o', markersize=4, label='Obs Pert: Post KL')
    ax.plot(steps / 1000, h_post_kl_means, 'm-s', markersize=4, label='H Pert: Post KL')
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Per-Row KL Divergence')
    ax.set_title('Comparison: Observation vs H Perturbation')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progression plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze smoothness with observation perturbation')
    parser.add_argument('--checkpoint_dir', '-c', type=str, default='',
                        help='Path to experiment folder containing a ckpt/ directory')
    parser.add_argument('--epsilon_min', type=float, default=1e-4,
                        help='Minimum epsilon as fraction of mean norm (e.g., 1e-4 = 0.01%)')
    parser.add_argument('--epsilon_max', type=float, default=1e-2,
                        help='Maximum epsilon as fraction of mean norm (e.g., 1e-2 = 1%)')
    parser.add_argument('--num_epsilon', type=int, default=10,
                        help='Number of epsilon values to test')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of state samples to collect (skipping terminal states)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default='smoothness_analysis.png',
                        help='Output plot filename')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID for rendering')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size for rendering')
    parser.add_argument('--unimix', type=float, default=0.01,
                        help='Uniform mixture coefficient for categorical distribution (default: 0.01)')
    parser.add_argument('--all_checkpoints', '-a', action='store_true',
                        help='Analyze all checkpoints in the ckpt directory')
    parser.add_argument('--checkpoint_step', type=int, default=None,
                        help='Specific checkpoint step to analyze (e.g., 100000)')
    parser.add_argument('--skip_steps', type=int, default=1,
                        help='Skip every N checkpoints when using --all_checkpoints (default: 1, analyze all)')

    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error('--checkpoint_dir is required')

    # Set up paths
    exp_dir = Path(args.checkpoint_dir)
    ckpt_dir = exp_dir / 'ckpt'
    config_path = exp_dir / 'config.yaml'

    print("="*80)
    print("DISTRIBUTION SMOOTHNESS ANALYSIS WITH OBSERVATION PERTURBATION")
    print("="*80)
    print(f"Experiment: {exp_dir.name}")
    print(f"Analysis approach:")
    print(f"  1. Collect trajectory with model states (h, z, prior, posterior, encoder output)")
    print(f"  2. For each sample, perturb observation: o2 = o1 + ε·noise")
    print(f"  3. Compare posterior distributions: q(z|h, e2) vs q(z|h, e1)")
    print(f"  4. Compute per-row KL for each of the 32 independent categoricals")
    print(f"  5. Also analyze h perturbation for Lipschitz properties")
    print(f"  Using unimix={args.unimix} for categorical distribution")
    print(f"  Epsilon = fraction of mean norm (min={args.epsilon_min:.2e}, max={args.epsilon_max:.2e})")
    print(f"    e.g., epsilon=0.01 means perturbation = 1% of mean norm")

    # Load config
    print("\nLoading configuration...")
    config = load_config(config_path)

    # Parse task name
    task_name = parse_task_from_folder(exp_dir.name, config)
    if task_name is None:
        print("ERROR: Could not determine task name!")
        return
    print(f"Task: {task_name}")

    # Set up JAX random key
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # Create epsilon fractions (as fraction of mean norm)
    epsilon_fractions = np.logspace(
        np.log10(args.epsilon_min),
        np.log10(args.epsilon_max),
        args.num_epsilon
    )

    # Create environment (shared across all checkpoints)
    env = create_environment(task_name)

    if args.all_checkpoints:
        # Analyze all checkpoints
        print("\n" + "="*80)
        print("ANALYZING ALL CHECKPOINTS")
        print("="*80)

        checkpoints = find_all_checkpoints(ckpt_dir)
        print(f"Found {len(checkpoints)} checkpoints")

        # Apply skip_steps
        if args.skip_steps > 1:
            checkpoints = checkpoints[::args.skip_steps]
            print(f"After skipping every {args.skip_steps} checkpoints: {len(checkpoints)} to analyze")

        # Print checkpoint list
        print("\nCheckpoints to analyze:")
        for step, path in checkpoints:
            print(f"  Step {step:>8}: {path.name}")

        # Create output directory for all checkpoint results
        output_dir = exp_dir / 'smoothness_analysis_all_ckpts'
        output_dir.mkdir(exist_ok=True)

        all_results = []
        for i, (step, ckpt_path) in enumerate(checkpoints):
            print(f"\n{'='*80}")
            print(f"Analyzing checkpoint {i+1}/{len(checkpoints)}: step {step}")
            print(f"{'='*80}")

            rng, key = jax.random.split(rng)
            result = analyze_single_checkpoint(
                ckpt_path, config, task_name, env, epsilon_fractions, args, key)

            if result is None:
                continue

            all_results.append(result)
            rng = result['rng']

            # Print brief summary
            prior_kl = result['direct_results']['prior_h_kl_mean'][-1]
            post_h_kl = result['direct_results']['post_h_kl_mean'][-1]
            post_e_kl = result['direct_results']['post_e_kl_mean'][-1]
            print(f"\n  Step {step} Summary (ε={epsilon_fractions[-1]:.2e} = {100*epsilon_fractions[-1]:.4f}% of mean norm):")
            print(f"    Prior KL (h pert): {prior_kl:.4e}")
            print(f"    Post KL (h pert): {post_h_kl:.4e}")
            print(f"    Post KL (e pert): {post_e_kl:.4e}")

            # Save individual checkpoint results
            step_output_path = output_dir / f'smoothness_step_{step:08d}.png'
            plot_results(result['state_results'], result['direct_results'], step_output_path)

            step_results_path = output_dir / f'smoothness_step_{step:08d}.npz'
            np.savez(step_results_path,
                     state_perturbation_results=result['state_results'],
                     direct_perturbation_results=result['direct_results'],
                     epsilon_fractions=epsilon_fractions,
                     task_name=task_name,
                     training_step=step,
                     num_samples=args.num_samples)

        # Close environment
        env.close()

        # Save combined results
        if len(all_results) > 1:
            combined_path = exp_dir / 'smoothness_all_checkpoints.npz'
            steps = [r['step'] for r in all_results]
            state_results_list = [r['state_results'] for r in all_results]
            direct_results_list = [r['direct_results'] for r in all_results]
            np.savez(combined_path,
                     steps=steps,
                     state_perturbation_results=state_results_list,
                     direct_perturbation_results=direct_results_list,
                     epsilon_fractions=epsilon_fractions,
                     task_name=task_name,
                     num_samples=args.num_samples)
            print(f"\nSaved combined results to: {combined_path}")

        print(f"\n{'='*80}")
        print(f"All checkpoints analyzed! Results saved to: {output_dir}")
        print(f"{'='*80}")

    elif args.checkpoint_step is not None:
        # Analyze specific checkpoint step
        checkpoints = find_all_checkpoints(ckpt_dir)
        ckpt_path = None
        for step, path in checkpoints:
            if step == args.checkpoint_step:
                ckpt_path = path
                break

        if ckpt_path is None:
            print(f"ERROR: Checkpoint at step {args.checkpoint_step} not found!")
            available_steps = [step for step, _ in checkpoints]
            print(f"Available steps: {available_steps}")
            env.close()
            return

        print(f"\nAnalyzing checkpoint at step {args.checkpoint_step}")

        rng, key = jax.random.split(rng)
        result = analyze_single_checkpoint(
            ckpt_path, config, task_name, env, epsilon_fractions, args, key)

        env.close()

        if result is None:
            return

        # Print summary
        print_summary(result['state_results'], result['direct_results'])

        # Generate plots with step in filename
        base_output = args.output.replace('.png', '')
        output_path = exp_dir / f'{base_output}_step_{args.checkpoint_step:08d}.png'
        plot_results(result['state_results'], result['direct_results'], output_path)

        # Save results
        results_path = exp_dir / f'smoothness_results_step_{args.checkpoint_step:08d}.npz'
        np.savez(results_path,
                 state_perturbation_results=result['state_results'],
                 direct_perturbation_results=result['direct_results'],
                 epsilon_fractions=epsilon_fractions,
                 task_name=task_name,
                 training_step=args.checkpoint_step,
                 num_samples=args.num_samples,
                 trajectory_h_states=result['trajectory']['h_states'],
                 trajectory_z_states=result['trajectory']['z_states'],
                 trajectory_prior_logits=result['trajectory']['prior_logits'],
                 trajectory_post_logits=result['trajectory']['post_logits'],
                 trajectory_encoder_outputs=result['trajectory']['encoder_outputs'],
                 trajectory_physics_states=result['trajectory']['physics_states'],
                 trajectory_actions=result['trajectory']['actions'],
                 trajectory_prev_actions=result['trajectory']['prev_actions'])
        print(f"\nSaved results to: {results_path}")

    else:
        # Original behavior: analyze latest checkpoint
        print("\nLoading checkpoint...")
        agent_data, step, ckpt_path = load_checkpoint(ckpt_dir)

        # Extract parameters
        params = None
        if agent_data is not None and isinstance(agent_data, dict):
            params = agent_data.get('params', None)
            if params is not None:
                print(f"\nExtracted {len(params)} parameter keys")
                dyn_keys = [k for k in params.keys() if k.startswith('dyn/')]
                enc_keys = [k for k in params.keys() if k.startswith('enc/')]
                print(f"  Dynamics parameters: {len(dyn_keys)}")
                print(f"  Encoder parameters: {len(enc_keys)}")

        if params is None:
            print("\nERROR: No parameters found in checkpoint!")
            env.close()
            return

        print(f"\nEpsilon fractions: [{args.epsilon_min:.2e}, {args.epsilon_max:.2e}] (as % of mean norm)")

        # Collect trajectory with model states
        rng, key = jax.random.split(rng)
        trajectory = collect_trajectory_with_model_states(
            env, params, config,
            num_samples=args.num_samples,
            camera_id=args.camera_id,
            size=(args.image_size, args.image_size),
            rng=key)

        # Run physics state perturbation analysis
        rng, key = jax.random.split(rng)
        state_results = analyze_physics_state_perturbation(
            env, params, config, trajectory, epsilon_fractions, key,
            camera_id=args.camera_id, size=(args.image_size, args.image_size),
            unimix=args.unimix)

        # Run direct perturbation analysis (h, e, a)
        rng, key = jax.random.split(rng)
        direct_results = analyze_direct_perturbation(
            params, config, trajectory, epsilon_fractions, key, unimix=args.unimix)

        # Close environment
        env.close()

        # Print summary
        print_summary(state_results, direct_results)

        # Generate plots
        output_path = exp_dir / args.output
        plot_results(state_results, direct_results, output_path)

        # Save results
        results_path = exp_dir / 'smoothness_results.npz'
        np.savez(results_path,
                 state_perturbation_results=state_results,
                 direct_perturbation_results=direct_results,
                 epsilon_fractions=epsilon_fractions,
                 task_name=task_name,
                 training_step=step,
                 num_samples=args.num_samples,
                 trajectory_h_states=trajectory['h_states'],
                 trajectory_z_states=trajectory['z_states'],
                 trajectory_prior_logits=trajectory['prior_logits'],
                 trajectory_post_logits=trajectory['post_logits'],
                 trajectory_encoder_outputs=trajectory['encoder_outputs'],
                 trajectory_physics_states=trajectory['physics_states'],
                 trajectory_actions=trajectory['actions'],
                 trajectory_prev_actions=trajectory['prev_actions'])
        print(f"\nSaved results to: {results_path}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
