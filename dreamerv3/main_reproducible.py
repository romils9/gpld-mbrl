import os

# --- START OF main.py ---
# This block MUST be at the very top, before all other imports.
# 1. SET THE JAX FLAGS
# We set these flags *before* JAX is imported for the first time.
print("[Reproducibility] Setting JAX environment flags...")
os.environ['XLA_FLAGS'] = (
    f'{os.environ.get("XLA_FLAGS", "")} --xla_gpu_deterministic_ops=true'
)
os.environ['JAX_DISABLE_MOST_FASTER_PATHS'] = '1'
print("--- Verifying Environment Flags ---")
print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS')}")
print(f"JAX_DISABLE_MOST_FASTER_PATHS: {os.environ.get('JAX_DISABLE_MOST_FASTER_PATHS')}")
print("-----------------------------------")

import importlib
import random
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml

from dreamerv3.utils.tqdm_logger import TqdmOutput  # <-- TQDM import


def main(argv=None):

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])

  for name in parsed.configs:
    config = config.update(configs[name])

  config = elements.Flags(config).parse(other)

  # ==================== ADDITION 1 ====================
  # Global seeding before importing Agent (so any import-time randomness is reproducible)
  random.seed(config.seed)
  np.random.seed(config.seed)
  # ====================================================

  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))

  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)

  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer
    # ==================== ADDITION 2 ====================
    # Worker-specific seeding so each portal worker gets its own deterministic stream
    worker_seed = 2 * int(config.seed) + int(config.replica)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # ====================================================

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)
  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)
  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)
  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)
  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)
  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)
  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)
  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent

  # ==================== ADDITION 3 ====================
  # Make sure any NumPy-based initialization inside make_agent is deterministic
  np.random.seed(config.seed)
  print(f'[Reproducibility] Agent seed: {config.seed}')
  # ====================================================

  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()

  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)

  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir

  agent_config = elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
      )

  return Agent(obs_space, act_space, agent_config)


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))

  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
      # Separate file for evaluation scores
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'evaluation_score.jsonl', 'eval_episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)

  # ---- TQDM progress bar (from Code 1) ----
  # Show a single progress bar on replica 0 if run steps are known.
  total_steps = (
      int(config.run.get('steps', 0))
      if isinstance(config.run, dict)
      else int(config.run.steps)
  )
  if total_steps > 0 and int(getattr(config, 'replica', 0)) == 0:
    desc = str(getattr(config, 'task', 'training'))
    outputs.append(TqdmOutput(total_steps, desc=desc))
  # -----------------------------------------

  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'

  # ==================== ADDITION 4 ====================
  # Deterministic seed per (folder, mode) combination.
  mode_int = 1 if mode == 'train' else 2
  folder_int = 10 if folder == 'replay' else 20
  replay_seed = int(config.seed) + mode_int + folder_int
  print(f'[Reproducibility] Replay {folder}/{mode} seed: {replay_seed}')
  # ====================================================

  kwargs = dict(
      length=length,
      capacity=int(capacity),
      online=config.replay.online,
      chunksize=config.replay.chunksize,
      directory=directory,
      seed=replay_seed,  # ADDED: replay RNG root seed
  )

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')

    recency = 1.0 / np.arange(1, int(capacity) + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors

    # Deterministic selectors using the replay_seed.
    kwargs['selector'] = selectors.Mixture(
        dict(
            uniform=selectors.Uniform(seed=replay_seed),
            priority=selectors.Prioritized(
                **config.replay.prio, seed=replay_seed + 1),
            recency=selectors.Recency(recency, seed=replay_seed + 2),
        ),
        config.replay.fracs,
        seed=replay_seed + 3,
    )

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa

  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]

  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)

  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)

  # ==================== MODIFICATION ====================
  # Always seed environments deterministically: seed + index
  env_seed = int(config.seed) + int(index)

  # Pass seed to environments that accept it
  if suite in {'atari', 'atari100k', 'dmc', 'procgen', 'crafter', 'dmlab'} or kwargs.pop('use_seed', False):
    kwargs['seed'] = env_seed
    print(f'[Reproducibility] Env {index} ({suite}) seed: {env_seed}')
  else:
    # For envs that don't take a seed, at least seed NumPy so their internal
    # NumPy-based randomness becomes deterministic.
    np.random.seed(env_seed)
    print(f'[Reproducibility] Env {index} ({suite}) numpy seed: {env_seed}')
  # ======================================================

  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'

  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)
  return stream


if __name__ == '__main__':
  main()
