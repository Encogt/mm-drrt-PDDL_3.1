#!/usr/bin/env python
"""Experiment sweep for Contribution 2 (motion-derived coordination-critical intervals):
how many duration-repair iterations the plan -> execute -> repair loop
(mm_drrt.pipeline_rai.run_pipeline) needs to converge, and how far the final durations end up
from the initial representative-sample estimate, over several seeds -- a mean and a range instead
of the single seed demos/demo2.py's walkthrough shows.

Same scenario and measurement calls as demos/demo2.py, with the step-by-step narration
(Walkthrough) dropped; see that file for the full explanation of what each number means and why
the arms execute at 60% of the planner's modelled speed (a deliberate model mismatch, so the
repair loop has something real to fix on every seed).

Usage:
    python experiments/run_contribution2.py                    # seeds 0-9 (default), headless
    python experiments/run_contribution2.py --seeds 0,1,2,3,4
    python experiments/run_contribution2.py --seed 0 --single   # one seed, printed, no sweep
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demos')))
from _walkthrough import pipeline_opt, setup_scene, close_scene  # noqa: E402

from mm_drrt import pipeline_rai  # noqa: E402

EXEC_SCALE = 0.6
MAX_RESOLVE_ATTEMPTS = 2


def run(seed):
    """One seed: measure representative (alpha, beta, duration) per action type, then
    plan/refine/execute/repair (mm_drrt.pipeline_rai.run_pipeline) at EXEC_SCALE x the planner's
    modelled joint speed, and return the convergence result."""
    opt = pipeline_opt(False, seed, '--env_type', 'exp_region_coordination_demo', '--num_robots', '2',
                       '--num_objs', '2', '--use_pddl_planner', '--region_mutex_enabled',
                       '--region_offsets_from_motion', '--durations_from_motion', '--region_offset_mode', 'two',
                       '--repair_strategy', 'resolve', '--max_resolve_attempts', str(MAX_RESOLVE_ATTEMPTS),
                       '--no_offsets_cache')
    C, env = setup_scene(opt)
    try:
        measured = env.measure_region_timing()
        planned = dict(measured)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        opt.execution_velocity_scale = EXEC_SCALE
        result = pipeline_rai.run_pipeline(env, opt, planned, 'experiment2')
        return {
            'seed': seed,
            'initial': {t: [round(float(x), 4) for x in v] for t, v in planned.items()},
            'final': {t: round(float(d), 4) for t, d in result.durations.items()},
            'effective': {t: [round(float(x), 4) for x in v] for t, v in result.effective.items()} if result.effective else {},
            'attempts': result.attempts,
            'converged': result.attempts <= MAX_RESOLVE_ATTEMPTS,
        }
    finally:
        close_scene(C)


def sweep(seeds, run_timeout):
    """Each seed as its own subprocess with a wall-clock cap, same isolation demos/demo4.py's
    sweep() uses -- a crash or timeout only loses that seed (reported and skipped)."""
    runs = {}
    for seed in seeds:
        tmp = tempfile.mkdtemp(prefix='contribution2_')
        out, log = os.path.join(tmp, 'result.json'), os.path.join(tmp, 'run.log')
        cmd = [sys.executable, os.path.abspath(__file__), '--seed', str(seed), '--single', '--json', out]
        start = time.time()
        with open(log, 'w') as f:
            try:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=run_timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
        r = json.load(open(out)) if os.path.exists(out) else {}
        if timed_out:
            r = {'status': f'timeout (> {run_timeout}s)'}
        elif 'attempts' not in r:
            lines = [ln.strip() for ln in open(log) if ln.strip()]
            why = r.get('error') or (lines[-1] if lines else 'no output')
            r = {'status': f'failed: {why}'}
        else:
            r['status'] = 'ok'
        runs[seed] = r
        msg = f"attempts={r['attempts']}  final={r['final']}" if r['status'] == 'ok' else ''
        print(f"  seed={seed}: {r['status']} in {time.time() - start:.0f}s  {msg}", flush=True)

    ok = [r for r in runs.values() if r['status'] == 'ok']
    print(f"\n{len(ok)}/{len(seeds)} seeds succeeded.")
    if not ok:
        return

    def stat(vals):
        return f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]" if len(vals) > 1 else f"{vals[0]:.3f}"

    print(f"  resolve attempts needed to converge: {stat([r['attempts'] for r in ok])}")
    print(f"  converged within the cap ({MAX_RESOLVE_ATTEMPTS}): {sum(r['converged'] for r in ok)}/{len(ok)}")
    for t in ('transit', 'transfer'):
        initial = [r['initial'][t][2] for r in ok if t in r['initial']]
        final = [r['final'][t] for r in ok if t in r['final']]
        if initial and final:
            print(f"  {t} duration: initial estimate {stat(initial)}s -> converged {stat(final)}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='Single run (used internally by the sweep)')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in range(10)),
                        help='Comma-separated seeds to sweep (default: 10 seeds, 0-9)')
    parser.add_argument('--run_timeout', type=int, default=600, help='Wall-clock cap (s) per seed')
    parser.add_argument('--json', type=str, default=None, help="Write this single run's result to a JSON file")
    parser.add_argument('--single', action='store_true', help='Run just --seed once instead of the --seeds sweep')
    args = parser.parse_args()

    if args.single:
        try:
            result = run(args.seed)
        except Exception as e:
            result = {'error': f"{type(e).__name__}: {e}"}
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(result, f)
        else:
            print(result)
        return

    sweep([int(x) for x in args.seeds.split(',')], args.run_timeout)


if __name__ == '__main__':
    main()
