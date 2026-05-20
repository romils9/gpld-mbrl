#!/usr/bin/env python3
"""
Process evaluation scores from DreamerV3/GPLD evaluation runs.

This is a variant of process_evaluation_scores.py that:
1) Parses Lipschitz logdir names: `_post{lam}_rows{N}_…` (rows right after post), or a longer tail where
   `_rows{N}_` may appear later (vision with rows), or no rows token (vision without rows → gp_rows=32).
2) Recursively discovers evaluation_score.jsonl files under the input directory.

Expected folder name examples:
- 20260415T153608_dmc_reacher_hard_lipschitz_prior0.0_post0.5_rows24_sqrt_time_h_25_seed4_tr512
- 20260413T225733_dmc_cheetah_run_lipschitz_prior0.0_post0.5_sqrt_time_gp_start_50000_h_25_seed0_tr512
- 20260415T153608_dmc_reacher_hard_baseline_h_25_seed4_tr512
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def _extract_rows_from_post_tail(tail: str):
  """
  Find the first `_rows{N}_` or `_rows{N}` / `rows{N}_` token in the string after `post{lam}_`.

  Returns:
    (gp_rows, decay_type): gp_rows is None if no rows token; decay_type omits the `_rowsN_` segment
    so aggregate filenames stay stable.
  """
  # Middle / leftmost: ..._rows{N}_...
  m = re.match(r'^(.*?)_rows(\d+)_(.+)$', tail)
  if m:
    before, n, after = m.group(1).strip('_'), int(m.group(2)), m.group(3).strip('_')
    parts = [p for p in (before, after) if p]
    decay = '_'.join(parts) if parts else 'none'
    return n, decay
  # Suffix: ..._rows{N}
  m = re.match(r'^(.*?)_rows(\d+)$', tail)
  if m:
    before = m.group(1).strip('_')
    n = int(m.group(2))
    decay = before if before else 'none'
    return n, decay
  # Prefix: rows{N}_... (unusual)
  m = re.match(r'^rows(\d+)_(.+)$', tail)
  if m:
    n = int(m.group(1))
    after = m.group(2).strip('_')
    decay = after if after else 'none'
    return n, decay
  return None, tail


def parse_folder_name(folder_name: str):
  """
  Parse folder name to extract metadata.

  Returns:
    dict with keys: timestamp, date, game, prior_lam, post_lam, gp_rows,
      decay_type, seed, h, size, tr, experiment_type
  """
  timestamp_match = re.match(r'^(\d{8}T\d{6})_', folder_name)
  if not timestamp_match:
    return None

  timestamp = timestamp_match.group(1)
  date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
  remainder = folder_name[len(timestamp) + 1:]

  size = '12m'
  size_match = re.search(r'_size(\d+m)$', remainder)
  if size_match:
    size = size_match.group(1)
    remainder = remainder[:size_match.start()]

  tr = 512
  tr_match = re.search(r'_tr(\d+)$', remainder)
  if tr_match:
    tr = int(tr_match.group(1))
    remainder = remainder[:tr_match.start()]

  seed_match = re.search(r'_seed(\d+)$', remainder)
  if not seed_match:
    return None
  seed = int(seed_match.group(1))
  remainder = remainder[:seed_match.start()]

  h = 15
  h_match = re.search(r'_h_(\d+)$', remainder)
  if h_match:
    h = int(h_match.group(1))
    remainder = remainder[:h_match.start()]

  if remainder.endswith('_baseline'):
    game = remainder[:-len('_baseline')]
    return {
      'timestamp': timestamp,
      'date': date,
      'game': game,
      'prior_lam': 0.0,
      'post_lam': 0.0,
      'gp_rows': 32,
      'decay_type': 'none',
      'seed': seed,
      'h': h,
      'size': size,
      'tr': tr,
      'experiment_type': 'baseline',
    }

  # Pattern:
  # {game}_lipschitz_prior{val}_post{val}_rows{N}[_{decay_type...}]
  lipschitz_match = re.match(
    r'^(.+?)_lipschitz_prior([\d.]+)_post([\d.]+)_rows(\d+)(?:_(.+))?$',
    remainder
  )
  if lipschitz_match:
    game = lipschitz_match.group(1)
    prior_lam = float(lipschitz_match.group(2))
    post_lam = float(lipschitz_match.group(3))
    gp_rows = int(lipschitz_match.group(4))
    decay_type = lipschitz_match.group(5) if lipschitz_match.group(5) else 'none'
    return {
      'timestamp': timestamp,
      'date': date,
      'game': game,
      'prior_lam': prior_lam,
      'post_lam': post_lam,
      'gp_rows': gp_rows,
      'decay_type': decay_type,
      'seed': seed,
      'h': h,
      'size': size,
      'tr': tr,
      'experiment_type': 'lipschitz',
    }

  # Vision-style and other layouts: `_post{lam}_` then an arbitrary tail that may or may not
  # contain `_rows{N}_` / `_rows{N}` / `rows{N}_` (rows not necessarily immediately after post).
  # Examples:
  #   ..._post0.5_sqrt_time_gp_start_50000
  #   ..._post0.5_rows16_sqrt_time_gp_start_50000   (also matched by strict pattern above)
  #   ..._post0.5_sqrt_time_rows16_gp_start_50000
  lipschitz_tail = re.match(
      r'^(.+?)_lipschitz_prior([\d.]+)_post([\d.]+)_(.+)$',
      remainder)
  if lipschitz_tail:
    game = lipschitz_tail.group(1)
    prior_lam = float(lipschitz_tail.group(2))
    post_lam = float(lipschitz_tail.group(3))
    tail = lipschitz_tail.group(4)
    n_rows, decay_type = _extract_rows_from_post_tail(tail)
    gp_rows = int(n_rows) if n_rows is not None else 32
    return {
        'timestamp': timestamp,
        'date': date,
        'game': game,
        'prior_lam': prior_lam,
        'post_lam': post_lam,
        'gp_rows': gp_rows,
        'decay_type': decay_type,
        'seed': seed,
        'h': h,
        'size': size,
        'tr': tr,
        'experiment_type': 'lipschitz',
    }

  return None


def read_evaluation_scores(jsonl_path: Path):
  scores = []
  with open(jsonl_path, 'r') as f:
    for line in f:
      data = json.loads(line.strip())
      step = data.get('step', 0)
      score = data.get('eval_episode/score', 0.0)
      scores.append([step, score])
  return np.array(scores)


def save_processed_data(output_dir: Path, metadata: dict, scores: np.ndarray):
  game_dir = output_dir / metadata['game']
  game_dir.mkdir(parents=True, exist_ok=True)

  filename = (
    f"prior{metadata['prior_lam']}_"
    f"post{metadata['post_lam']}_"
    f"rows{metadata['gp_rows']}_"
    f"{metadata['decay_type']}_"
    f"h{metadata['h']}_"
    f"size{metadata['size']}_"
    f"tr{metadata['tr']}_"
    f"seed{metadata['seed']}_"
    f"{metadata['date']}.npz"
  )

  filepath = game_dir / filename
  np.savez(
    filepath,
    scores=scores,
    step=scores[:, 0],
    score=scores[:, 1],
    prior_lam=metadata['prior_lam'],
    post_lam=metadata['post_lam'],
    gp_rows=metadata['gp_rows'],
    decay_type=metadata['decay_type'],
    seed=metadata['seed'],
    date=metadata['date'],
    game=metadata['game'],
    timestamp=metadata['timestamp'],
    h=metadata['h'],
    size=metadata['size'],
    tr=metadata['tr'],
  )

  csv_path = filepath.with_suffix('.csv')
  with open(csv_path, 'w') as f:
    f.write(f"# Game: {metadata['game']}\n")
    f.write(f"# Prior Lambda: {metadata['prior_lam']}\n")
    f.write(f"# Post Lambda: {metadata['post_lam']}\n")
    f.write(f"# GP Rows: {metadata['gp_rows']}\n")
    f.write(f"# Decay Type: {metadata['decay_type']}\n")
    f.write(f"# Imagination Horizon (h): {metadata['h']}\n")
    f.write(f"# Model Size: {metadata['size']}\n")
    f.write(f"# Train Ratio: {metadata['tr']}\n")
    f.write(f"# Seed: {metadata['seed']}\n")
    f.write(f"# Date: {metadata['date']}\n")
    f.write(f"# Timestamp: {metadata['timestamp']}\n")
    f.write("step,score\n")
    for row in scores:
      f.write(f"{int(row[0])},{row[1]}\n")

  return filepath


def compute_seed_ranking(seed_data, num_last_scores=5):
  seed_rankings = []
  for sd in seed_data:
    vals = sd['scores'][:, 1]
    avg_last = np.mean(vals[-num_last_scores:]) if len(vals) >= num_last_scores else np.mean(vals)
    seed_rankings.append((sd['seed'], avg_last, sd))
  seed_rankings.sort(key=lambda x: x[1], reverse=True)
  return seed_rankings


def get_iqm_seeds(seed_rankings):
  n = len(seed_rankings)
  if n < 4:
    return [sr[2] for sr in seed_rankings]
  lower_idx = n // 4
  upper_idx = n - (n // 4)
  return [sr[2] for sr in seed_rankings[lower_idx:upper_idx]]


def compute_mean_across_seeds(output_dir: Path):
  results = {}
  for game_dir in output_dir.iterdir():
    if not game_dir.is_dir():
      continue
    game = game_dir.name
    results[game] = {}
    config_data = defaultdict(list)

    for npz_file in game_dir.glob("*.npz"):
      if npz_file.name.startswith(("MEAN_", "MEDIAN_", "IQM_")):
        continue
      data = np.load(npz_file, allow_pickle=True)
      prior_lam = float(data['prior_lam'])
      post_lam = float(data['post_lam'])
      gp_rows = int(data['gp_rows']) if 'gp_rows' in data else 32
      decay_type = str(data['decay_type'])
      h = int(data['h']) if 'h' in data else 15
      size = str(data['size']) if 'size' in data else '12m'
      tr = int(data['tr']) if 'tr' in data else 512
      scores = data['scores']
      seed = int(data['seed'])

      config_key = (prior_lam, post_lam, gp_rows, decay_type, h, size, tr)
      config_data[config_key].append({'seed': seed, 'scores': scores})

    for config_key, seed_data in config_data.items():
      prior_lam, post_lam, gp_rows, decay_type, h, size, tr = config_key

      seed_rankings = compute_seed_ranking(seed_data, num_last_scores=5)
      iqm_seed_data = get_iqm_seeds(seed_rankings)
      iqm_seeds = [sd['seed'] for sd in iqm_seed_data]

      all_steps = set()
      for sd in seed_data:
        all_steps.update(sd['scores'][:, 0].astype(int))
      all_steps = sorted(all_steps)

      seed_scores = {}
      for sd in seed_data:
        seed = sd['seed']
        seed_scores[seed] = {int(s[0]): s[1] for s in sd['scores']}

      mean_scores, std_scores, median_scores, num_seeds_per_step = [], [], [], []
      for step in all_steps:
        vals = [seed_scores[s][step] for s in seed_scores if step in seed_scores[s]]
        if vals:
          mean_scores.append([step, np.mean(vals)])
          std_scores.append([step, np.std(vals)])
          median_scores.append([step, np.median(vals)])
          num_seeds_per_step.append([step, len(vals)])

      mean_scores = np.array(mean_scores)
      std_scores = np.array(std_scores)
      median_scores = np.array(median_scores)
      num_seeds_per_step = np.array(num_seeds_per_step)

      iqm_scores, iqm_std_scores, iqm_num_seeds_per_step = [], [], []
      for step in all_steps:
        vals = [seed_scores[s][step] for s in iqm_seeds if step in seed_scores.get(s, {})]
        if vals:
          iqm_scores.append([step, np.mean(vals)])
          iqm_std_scores.append([step, np.std(vals)])
          iqm_num_seeds_per_step.append([step, len(vals)])

      iqm_scores = np.array(iqm_scores) if iqm_scores else np.array([[0, 0]])
      iqm_std_scores = np.array(iqm_std_scores) if iqm_std_scores else np.array([[0, 0]])
      iqm_num_seeds_per_step = np.array(iqm_num_seeds_per_step) if iqm_num_seeds_per_step else np.array([[0, 0]])

      config_name = f"prior{prior_lam}_post{post_lam}_rows{gp_rows}_{decay_type}_h{h}_size{size}_tr{tr}"

      # Save MEAN
      np.savez(
        game_dir / f"MEAN_{config_name}.npz",
        mean_scores=mean_scores,
        std_scores=std_scores,
        num_seeds_per_step=num_seeds_per_step,
        prior_lam=prior_lam,
        post_lam=post_lam,
        gp_rows=gp_rows,
        decay_type=decay_type,
        h=h,
        size=size,
        tr=tr,
        num_seeds=len(seed_data),
        seeds=[sd['seed'] for sd in seed_data],
      )
      with open(game_dir / f"MEAN_{config_name}.csv", 'w') as f:
        f.write(f"# Game: {game}\n")
        f.write(f"# Prior Lambda: {prior_lam}\n")
        f.write(f"# Post Lambda: {post_lam}\n")
        f.write(f"# GP Rows: {gp_rows}\n")
        f.write(f"# Decay Type: {decay_type}\n")
        f.write(f"# Imagination Horizon (h): {h}\n")
        f.write(f"# Model Size: {size}\n")
        f.write(f"# Train Ratio: {tr}\n")
        f.write(f"# Number of Seeds: {len(seed_data)}\n")
        f.write(f"# Seeds: {[sd['seed'] for sd in seed_data]}\n")
        f.write("step,mean_score,std_score,num_seeds\n")
        for i in range(len(mean_scores)):
          f.write(f"{int(mean_scores[i, 0])},{mean_scores[i, 1]},{std_scores[i, 1]},{int(num_seeds_per_step[i, 1])}\n")

      # Save MEDIAN
      np.savez(
        game_dir / f"MEDIAN_{config_name}.npz",
        median_scores=median_scores,
        num_seeds_per_step=num_seeds_per_step,
        prior_lam=prior_lam,
        post_lam=post_lam,
        gp_rows=gp_rows,
        decay_type=decay_type,
        h=h,
        size=size,
        tr=tr,
        num_seeds=len(seed_data),
        seeds=[sd['seed'] for sd in seed_data],
      )
      with open(game_dir / f"MEDIAN_{config_name}.csv", 'w') as f:
        f.write(f"# Game: {game}\n")
        f.write("# Type: MEDIAN\n")
        f.write(f"# Prior Lambda: {prior_lam}\n")
        f.write(f"# Post Lambda: {post_lam}\n")
        f.write(f"# GP Rows: {gp_rows}\n")
        f.write(f"# Decay Type: {decay_type}\n")
        f.write(f"# Imagination Horizon (h): {h}\n")
        f.write(f"# Model Size: {size}\n")
        f.write(f"# Train Ratio: {tr}\n")
        f.write(f"# Number of Seeds: {len(seed_data)}\n")
        f.write(f"# Seeds: {[sd['seed'] for sd in seed_data]}\n")
        f.write("step,median_score,num_seeds\n")
        for i in range(len(median_scores)):
          f.write(f"{int(median_scores[i, 0])},{median_scores[i, 1]},{int(num_seeds_per_step[i, 1])}\n")

      # Save IQM
      np.savez(
        game_dir / f"IQM_{config_name}.npz",
        iqm_scores=iqm_scores,
        iqm_std_scores=iqm_std_scores,
        iqm_num_seeds_per_step=iqm_num_seeds_per_step,
        prior_lam=prior_lam,
        post_lam=post_lam,
        gp_rows=gp_rows,
        decay_type=decay_type,
        h=h,
        size=size,
        tr=tr,
        num_seeds=len(seed_data),
        all_seeds=[sd['seed'] for sd in seed_data],
        iqm_seeds=iqm_seeds,
        seed_rankings=[(sr[0], sr[1]) for sr in seed_rankings],
      )
      with open(game_dir / f"IQM_{config_name}.csv", 'w') as f:
        f.write(f"# Game: {game}\n")
        f.write("# Type: IQM (Interquartile Mean - middle 50% of seeds)\n")
        f.write(f"# Prior Lambda: {prior_lam}\n")
        f.write(f"# Post Lambda: {post_lam}\n")
        f.write(f"# GP Rows: {gp_rows}\n")
        f.write(f"# Decay Type: {decay_type}\n")
        f.write(f"# Imagination Horizon (h): {h}\n")
        f.write(f"# Model Size: {size}\n")
        f.write(f"# Train Ratio: {tr}\n")
        f.write(f"# Total Seeds: {len(seed_data)}\n")
        f.write(f"# All Seeds (ranked by avg of last 5 scores): {[(sr[0], f'{sr[1]:.2f}') for sr in seed_rankings]}\n")
        f.write(f"# IQM Seeds (middle 50%): {iqm_seeds}\n")
        f.write("step,iqm_score,iqm_std,num_seeds\n")
        for i in range(len(iqm_scores)):
          f.write(f"{int(iqm_scores[i, 0])},{iqm_scores[i, 1]},{iqm_std_scores[i, 1]},{int(iqm_num_seeds_per_step[i, 1])}\n")

      results[game][config_name] = {
        'prior_lam': prior_lam,
        'post_lam': post_lam,
        'gp_rows': gp_rows,
        'decay_type': decay_type,
        'h': h,
        'size': size,
        'tr': tr,
        'num_seeds': len(seed_data),
        'seeds': [sd['seed'] for sd in seed_data],
        'seed_rankings': [(sr[0], sr[1]) for sr in seed_rankings],
        'iqm_seeds': iqm_seeds,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'median_scores': median_scores,
        'iqm_scores': iqm_scores,
        'iqm_std_scores': iqm_std_scores,
        'num_seeds_per_step': num_seeds_per_step,
      }

  return results


def main():
  parser = argparse.ArgumentParser(description="Process eval scores for post-ICML runs (rows-aware).")
  parser.add_argument(
    '--input', '-i', type=str,
    default='./evals',
    help='Root directory to recursively search for evaluation_score.jsonl files.'
  )
  parser.add_argument(
    '--output', '-o', type=str,
    default='./processed_scores',
    help='Output directory for processed per-game scores and aggregate stats.'
  )
  args = parser.parse_args()

  input_dir = Path(args.input)
  output_dir = Path(args.output)
  output_dir.mkdir(parents=True, exist_ok=True)

  processed_count = 0
  skipped_count = 0

  for eval_file in sorted(input_dir.rglob('evaluation_score.jsonl')):
    folder = eval_file.parent
    metadata = parse_folder_name(folder.name)
    if metadata is None:
      skipped_count += 1
      continue
    try:
      scores = read_evaluation_scores(eval_file)
    except Exception:
      skipped_count += 1
      continue
    if len(scores) == 0:
      skipped_count += 1
      continue
    save_processed_data(output_dir, metadata, scores)
    processed_count += 1

  print(f"Processed {processed_count} experiments, skipped {skipped_count}")
  compute_mean_across_seeds(output_dir)
  print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
  main()
