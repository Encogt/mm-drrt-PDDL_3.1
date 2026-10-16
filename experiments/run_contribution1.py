#!/usr/bin/env python
"""Experiment sweep for Contribution 1 (no given plan skeleton: PDDL 2.1 planning + unchanged
MM-dRRT refinement): planning/refinement timing and whether Tamer's derived skeleton matches the
hand-written one, for both the one-box and two-box crossing relay (exp_two_robots_rai), over
several seeds -- a mean and a range instead of the single seed demos/demo1.py's walkthrough shows.

Reuses demos/demo1.py's own same_skeleton() directly, with the step-by-step narration
(Walkthrough) and replay dropped; see that file for the full explanation of what each number
means. The one-box scenario has no hand-written skeleton to compare against (demo1.py only shows
that comparison for the two-box case), so 'skeleton_matches_handwritten' is only recorded there.

Usage:
    python experiments/run_contribution1.py                    # seeds 0-9 (default), both scenarios
    python experiments/run_contribution1.py --seeds 0,1,2,3,4
    python experiments/run_contribution1.py --seed 0 --num_objs 2 --single   # one run, printed
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demos')))
from _walkthrough import pipeline_opt, setup_scene, close_scene  # noqa: E402
from demo1 import same_skeleton  # noqa: E402

from mm_drrt.pipeline_rai import solve_task_plan, refine  # noqa: E402


def run(seed, num_objs):
    """One (seed, num_objs) scenario: Tamer derives the skeleton from the PDDL problem alone, the
    SAME unchanged PlanSkeleton/dRRT* refines it, and (two-box case only) the derived skeleton is
    compared against the environment's hand-written one."""
    opt = pipeline_opt(False, seed, '--env_type', 'exp_two_robots_rai', '--num_robots', '2',
                       '--num_objs', str(num_objs), '--use_pddl_planner', '--no_offsets_cache')
    C, env = setup_scene(opt)
    try:
        t0 = time.time()
        plan, action_orders, obj_orders, constraints, used, schedule, _, _ = solve_task_plan(env, opt, None)
        plan_secs = time.time() - t0
        handoffs = [c for c in constraints if plan[c['pre']][2] == plan[c['post']][2]]

        out = {
            'seed': seed,
            'num_objs': num_objs,
            'planner_used': used,
            'plan_secs': round(plan_secs, 4),
            'makespan': round(max(a['end'] for a in schedule), 4) if schedule else None,
            'num_actions': len(plan),
        }
        # create_plan_order_constraints() has a hand-written skeleton for BOTH num_objs=1 and
        # num_objs=2 (examples/envs/example_two_robots_rai_env.py) -- demos/demo1.py's walkthrough
        # only narrates the comparison for num_objs=2, but the data exists for num_objs=1 too.
        manual = env.create_plan_order_constraints()
        out['skeleton_matches_handwritten'] = same_skeleton(manual, (plan, action_orders, obj_orders, handoffs))

        composite_path, refine_secs = refine(env, opt, plan, obj_orders, constraints)
        out['refine_secs'] = round(refine_secs, 4)
        out['composite_path_nodes'] = len(composite_path)
        return out
    finally:
        close_scene(C)


def sweep(seeds, num_objs_list, run_timeout):
    """Each (num_objs, seed) as its own subprocess with a wall-clock cap, same isolation
    demos/demo4.py's sweep() uses -- a crash or timeout only loses that run (reported and
    skipped)."""
    runs = {}
    for num_objs in num_objs_list:
        for seed in seeds:
            tmp = tempfile.mkdtemp(prefix='contribution1_')
            out, log = os.path.join(tmp, 'result.json'), os.path.join(tmp, 'run.log')
            cmd = [sys.executable, os.path.abspath(__file__), '--seed', str(seed), '--num_objs', str(num_objs),
                  '--single', '--json', out]
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
            elif 'plan_secs' not in r:
                lines = [ln.strip() for ln in open(log) if ln.strip()]
                why = r.get('error') or (lines[-1] if lines else 'no output')
                r = {'status': f'failed: {why}'}
            else:
                r['status'] = 'ok'
            runs[(num_objs, seed)] = r
            msg = (f"plan={r['plan_secs']:.3f}s  makespan={r['makespan']}  refine={r['refine_secs']:.2f}s  "
                  + (f"match={r['skeleton_matches_handwritten']}" if 'skeleton_matches_handwritten' in r else '')
                  ) if r['status'] == 'ok' else ''
            print(f"  num_objs={num_objs} seed={seed}: {r['status']} in {time.time() - start:.0f}s  {msg}", flush=True)

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return '-'
        return f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]" if len(vals) > 1 else f"{vals[0]:.3f}"

    for num_objs in num_objs_list:
        ok = [runs[(num_objs, s)] for s in seeds if runs[(num_objs, s)]['status'] == 'ok']
        label = 'two-box crossing relay' if num_objs == 2 else 'one-box relay'
        print(f"\n{label} (num_objs={num_objs}): {len(ok)}/{len(seeds)} seeds succeeded.")
        if not ok:
            continue
        print(f"  Tamer planning time:   {stat([r['plan_secs'] for r in ok])}s")
        print(f"  Tamer's planned makespan: {stat([r['makespan'] for r in ok])}")
        print(f"  dRRT* refinement time: {stat([r['refine_secs'] for r in ok])}s")
        print(f"  composite path nodes:  {stat([r['composite_path_nodes'] for r in ok])}")
        matches = [r['skeleton_matches_handwritten'] for r in ok]
        print(f"  derived skeleton matches hand-written one: {sum(matches)}/{len(ok)} seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='Single run (used internally by the sweep)')
    parser.add_argument('--num_objs', type=int, default=2, help='Single run: 1 (one-box relay) or 2 (two-box crossing relay)')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in range(10)),
                        help='Comma-separated seeds to sweep (default: 10 seeds, 0-9)')
    parser.add_argument('--num_objs_list', type=str, default='1,2',
                        help='Comma-separated num_objs scenarios to sweep (default: both 1 and 2)')
    parser.add_argument('--run_timeout', type=int, default=600, help='Wall-clock cap (s) per run')
    parser.add_argument('--json', type=str, default=None, help="Write this single run's result to a JSON file")
    parser.add_argument('--single', action='store_true', help='Run just --seed/--num_objs once instead of the sweep')
    args = parser.parse_args()

    if args.single:
        try:
            result = run(args.seed, args.num_objs)
        except Exception as e:
            result = {'error': f"{type(e).__name__}: {e}"}
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(result, f)
        else:
            print(result)
        return

    sweep([int(x) for x in args.seeds.split(',')], [int(x) for x in args.num_objs_list.split(',')], args.run_timeout)


if __name__ == '__main__':
    main()
