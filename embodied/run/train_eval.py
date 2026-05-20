import collections
import os
import pickle
import shutil
from functools import partial as bind

import elements
import embodied
import numpy as np


def train_eval(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  agent = make_agent()
  replay_train = make_replay_train()
  replay_eval = make_replay_eval()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  train_episodes = collections.defaultdict(elements.Agg)
  train_epstats = elements.Agg()
  eval_episodes = collections.defaultdict(elements.Agg)
  eval_epstats = elements.Agg()
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = elements.when.Clock(args.log_every)
  should_save = elements.when.Clock(args.save_every)

  # Step-based evaluation trigger using eval_freq
  eval_freq = getattr(args, 'eval_freq', 5000)
  last_eval_step = [0]  # Use list to allow mutation in closure
  def should_eval(current_step):
    if eval_freq <= 0:
      return False
    current = int(current_step)
    if current >= last_eval_step[0] + eval_freq:
      last_eval_step[0] = (current // eval_freq) * eval_freq
      return True
    return False

  # Step-based checkpoint saving (in addition to time-based)
  # save_every_steps: if > 0, save checkpoint every N steps
  # save_on_eval: if True, save checkpoint after each evaluation
  # keep_checkpoints: how many step-based checkpoints to keep (0 = keep all)
  save_every_steps = getattr(args, 'save_every_steps', 0)
  save_on_eval = getattr(args, 'save_on_eval', True)  # Default: save on every eval
  keep_checkpoints = getattr(args, 'keep_checkpoints', 0)  # 0 = keep all
  last_save_step = [0]
  def should_save_step(current_step):
    if save_every_steps <= 0:
      return False
    current = int(current_step)
    if current >= last_save_step[0] + save_every_steps:
      last_save_step[0] = (current // save_every_steps) * save_every_steps
      return True
    return False

  # Multi-checkpoint saving to step-numbered folders
  ckpt_dir = logdir / 'ckpt'
  ckpt_dir.mkdir()

  def save_checkpoint_at_step(current_step, agent_obj, replay_train_obj, replay_eval_obj, tag=''):
    """Save checkpoint to a step-numbered folder."""
    step_int = int(current_step)
    folder_name = f'step_{step_int:08d}'
    if tag:
      folder_name = f'{folder_name}_{tag}'
    ckpt_path = ckpt_dir / folder_name
    ckpt_path.mkdir()

    # Save agent state
    agent_state = agent_obj.save()
    with open(str(ckpt_path / 'agent.pkl'), 'wb') as f:
      pickle.dump(agent_state, f)

    # Save step
    with open(str(ckpt_path / 'step.pkl'), 'wb') as f:
      pickle.dump(step_int, f)

    # Save replay metadata (not the actual data, just like elements.Checkpoint)
    with open(str(ckpt_path / 'replay_train.pkl'), 'wb') as f:
      pickle.dump(None, f)
    with open(str(ckpt_path / 'replay_eval.pkl'), 'wb') as f:
      pickle.dump(None, f)

    # Mark as done
    (ckpt_path / 'done').write('')

    # Update latest pointer
    (ckpt_dir / 'latest').write(folder_name)

    print(f'  Saved checkpoint to {ckpt_path}')

    # Clean up old checkpoints if keep_checkpoints > 0
    if keep_checkpoints > 0:
      cleanup_old_checkpoints(keep_checkpoints)

  def cleanup_old_checkpoints(keep_n):
    """Keep only the N most recent step-based checkpoints."""
    # Find all step_* folders
    step_folders = []
    for item in os.listdir(str(ckpt_dir)):
      item_path = ckpt_dir / item
      if item.startswith('step_') and os.path.isdir(str(item_path)):
        # Extract step number
        try:
          step_num = int(item.split('_')[1])
          step_folders.append((step_num, item))
        except (ValueError, IndexError):
          continue

    # Sort by step number (descending) and remove old ones
    step_folders.sort(key=lambda x: x[0], reverse=True)
    folders_to_remove = step_folders[keep_n:]

    for _, folder_name in folders_to_remove:
      folder_path = ckpt_dir / folder_name
      try:
        shutil.rmtree(str(folder_path))
        print(f'  Removed old checkpoint: {folder_name}')
      except Exception as e:
        print(f'  Warning: Could not remove {folder_name}: {e}')

  @elements.timer.section('logfn')
  def logfn(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      # Use different prefix for eval vs train episodes
      prefix = 'eval_episode' if mode == 'eval' else 'episode'
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix=prefix)
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env_train, i) for i in range(args.envs)]
  driver_train = embodied.Driver(fns, parallel=(not args.debug))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(lambda tran, _: policy_fps.step())
  driver_train.on_step(replay_train.add)
  driver_train.on_step(bind(logfn, mode='train'))

  fns = [bind(make_env_eval, i) for i in range(args.eval_envs)]
  driver_eval = embodied.Driver(fns, parallel=(not args.debug))
  driver_eval.on_step(replay_eval.add)
  driver_eval.on_step(bind(logfn, mode='eval'))
  driver_eval.on_step(lambda tran, _: policy_fps.step())

  stream_train = iter(agent.stream(make_stream(replay_train, 'train')))
  stream_report = iter(agent.stream(make_stream(replay_train, 'report')))
  stream_eval = iter(agent.stream(make_stream(replay_eval, 'eval')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  carry_eval = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay_train) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay_train.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver_train.on_step(trainfn)

  def reportfn(carry, stream):
    agg = elements.Agg()
    for _ in range(args.report_batches):
      batch = next(stream)
      carry, mets = agent.report(carry, batch)
      agg.add(mets)
    return carry, agg.result()

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay_train = replay_train
  cp.replay_eval = replay_eval
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  driver_train.reset(agent.init_policy)
  while step < args.steps:

    if should_eval(step):
      print(f'Evaluation at step {int(step)}')
      driver_eval.reset(agent.init_policy)
      driver_eval(eval_policy, episodes=args.eval_eps)
      logger.add(eval_epstats.result(), prefix='eval_epstats')
      if len(replay_train):
        carry_report, mets = reportfn(carry_report, stream_report)
        logger.add(mets, prefix='report')
      if len(replay_eval):
        carry_eval, mets = reportfn(carry_eval, stream_eval)
        logger.add(mets, prefix='eval')
      # Save checkpoint after evaluation if save_on_eval is enabled
      if save_on_eval:
        print(f'Saving checkpoint at step {int(step)} (post-eval)')
        save_checkpoint_at_step(step, agent, replay_train, replay_eval, tag='eval')

    driver_train(train_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(replay_train.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

    # Step-based checkpoint saving (independent of time-based)
    if should_save_step(step):
      print(f'Saving checkpoint at step {int(step)} (step-based)')
      save_checkpoint_at_step(step, agent, replay_train, replay_eval, tag='step')

  # Final checkpoint save at end of training
  print(f'Saving final checkpoint at step {int(step)}')
  save_checkpoint_at_step(step, agent, replay_train, replay_eval, tag='final')
  logger.close()
