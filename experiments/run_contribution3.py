#!/usr/bin/env python
"""Experiment sweep for Contribution 3 (least-commitment conflict resolution): whether the
minimal-interval schedule is actually least-commitment, how much shorter it is than a full-mutex
serialization, and how much the per-(robot, region) fluent mode's cheaper-order choice saves --
a mean and a range over several seeds instead of the single seed demos/demo3.py's walkthrough
shows.

Reuses demos/demo3.py's own solve()/pad_overlap() helpers and scenario directly, with the
step-by-step narration (Walkthrough) and replay dropped; see that file for the full explanation
of what each number means.

Usage:
    python experiments/run_contribution3.py                    # seeds 0-9 (default), headless
    python experiments/run_contribution3.py --seeds 0,1,2,3,4
    python experiments/run_contribution3.py --seed 0 --single   # one seed, printed, no sweep
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

from demo3 import solve, pad_overlap  # noqa: E402

from mm_drrt.pipeline_rai import refine, planner_settings, handoff_constraints_only  # noqa: E402
from mm_drrt.planner.tamer_pddl_planner import _solve_with_wall_clock_timeout  # noqa: E402
from mm_drrt.planner.pddl_problem_generator import generate_problem  # noqa: E402
from mm_drrt.utils.schedule_repair import check_least_commitment, is_least_commitment, offsets_by_action, \
    retime_schedule  # noqa: E402
from mm_drrt.utils.pddl_parser import parse_pddl_plan  # noqa: E402
from mm_drrt.utils.minimal_temporal_constraint import derive_minimal_constraint  # noqa: E402
from mm_drrt.utils.motion_timing import velocity_limits, executed_action_timeline, async_schedule  # noqa: E402
from mm_drrt.utils.rai_motion_planner_utils import sample_region_offsets  # noqa: E402
from mm_drrt.utils import rai_utils as ru  # noqa: E402
from compare_region_fluents_demo import _TransferOnlyProblem  # noqa: E402


def run(seed):
    """One seed: solve stock/full-mutex/minimal, check least-commitment, re-time the SAME refined
    motion two ways (full mutex vs minimal), and compare the per-(robot, region) fluent mode's two
    possible orders. Mirrors demos/demo3.py's walkthrough, data only."""
    opt = pipeline_opt(False, seed, '--env_type', 'exp_region_coordination_demo', '--num_robots', '2',
                       '--num_objs', '2', '--use_pddl_planner', '--region_mutex_enabled',
                       '--region_offsets_from_motion', '--durations_from_motion', '--region_offset_mode', 'two',
                       '--no_offsets_cache')
    C, env = setup_scene(opt)
    initial_world = env.save_world()
    out = {'seed': seed}
    try:
        measured = env.measure_region_timing()
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        durations = dict(transit_duration=measured['transit'][2], transfer_duration=measured['transfer'][2])

        stock = solve(env, 'stock', **durations)
        full = solve(env, 'full mutex', region_mutex_enabled=True, **durations)
        kwargs, effective, _ = planner_settings(env, opt, measured, use_four=False)
        kwargs.pop('timeout')
        minimal = solve(env, 'minimal', **kwargs)
        out['stock_pad_overlap'] = pad_overlap(stock['schedule'], measured)
        out['full_makespan'] = full['makespan']
        out['minimal_makespan'] = minimal['makespan']
        out['minimal_vs_full_pct'] = 100 * (1 - minimal['makespan'] / full['makespan'])

        checks = check_least_commitment(minimal['schedule'], offsets_by_action(minimal['schedule'], effective))
        out['least_commitment'] = is_least_commitment(checks)
        out['pairs'] = [{'pair': f"{c.first}->{c.second}", 'actual_gap': round(c.actual_gap, 4),
                         'required_gap': round(c.required_gap, 4), 'minimal_gap': round(c.minimal_gap, 4),
                         'minimal_order_chosen': c.minimal_order_chosen} for c in checks]

        # Same refined motion, two coordination rules (full mutex vs minimal), re-timed.
        env.restore_world(initial_world)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        plan = minimal['plan']
        path, _secs = refine(env, opt, plan, minimal['obj_orders'],
                             handoff_constraints_only(plan, minimal['constraints']))
        timeline = executed_action_timeline(env, path, minimal['action_orders'], plan)
        sched = minimal['schedule']
        pad_pairs = [(x['name'], y['name'], 0.0, timeline[x['name']]['duration'], 0.0, timeline[y['name']]['duration'])
                     for i, x in enumerate(sched) for y in sched[i + 1:]
                     if x['name'] in timeline and y['name'] in timeline and x['robot'] != y['robot']
                     and x['region'] == y['region']]
        async_full = async_schedule(env, sched, timeline, conflicts=pad_pairs)
        async_minimal = async_schedule(env, sched, timeline, conflicts=None)
        out['async_full_makespan'] = async_full.makespan if async_full.feasible else None
        out['async_minimal_makespan'] = async_minimal.makespan if async_minimal.feasible else None
        if async_full.feasible and async_minimal.feasible:
            out['async_savings'] = async_full.makespan - async_minimal.makespan

        # Per-(robot, region) fluent mode: each robot keeps its own interval; the cheaper order.
        env.restore_world(initial_world)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        per_robot = {}
        for r, key in ((0, 'r0'), (1, 'r1')):
            robot = env.robots[r]
            saved = env.save_world()
            res = sample_region_offsets(robot, env._arm, env._grasp_type, env.m_objs[r], env.f_objs[2], 'transfer',
                                        None, collision_objs=env.fixed_obstacles,
                                        velocity_limits=velocity_limits(ru._spec_of(robot)[0]))
            env.restore_world(saved)
            per_robot[key] = res
        duration = max(per_robot['r0'][2], per_robot['r1'][2])
        region_offsets = {(k, 'drop_pad'): {'transfer': v[:2]} for k, v in per_robot.items()}
        problem, mapper = generate_problem(_TransferOnlyProblem(region_offsets), transit_duration=duration,
                                           transfer_duration=duration, region_mutex_enabled=True,
                                           use_region_fluents=True, region_offsets=region_offsets)
        status, solved = _solve_with_wall_clock_timeout(problem, 60)
        _plan, _, _, _, schedule = parse_pddl_plan(solved, mapper, env, return_schedule=True)
        offsets = {a['name']: per_robot[a['robot']][:2] for a in schedule}
        schedule = [dict(a, start=a['occupy_start'] - offsets[a['name']][0],
                         end=a['occupy_start'] - offsets[a['name']][0] + duration) for a in schedule]
        mc = derive_minimal_constraint(*per_robot['r0'][:2], *per_robot['r1'][:2])
        cheaper = 'r0' if mc.order == 'i_before_j' else 'r1'
        first = min(schedule, key=lambda a: a['occupy_start'])
        [c] = check_least_commitment(schedule, offsets)
        durs = {a['name']: duration for a in schedule}
        minimal_r = retime_schedule(schedule, offsets, durs, order='minimal')
        maximal_r = retime_schedule(schedule, offsets, durs, order='maximal')
        out['fluent_cheaper_order'] = cheaper
        out['fluent_tamer_chose_cheaper'] = first['robot'] == cheaper
        out['fluent_tamer_order_least_commitment'] = c.minimal_order_chosen
        out['fluent_cheaper_makespan'] = minimal_r.makespan
        out['fluent_other_makespan'] = maximal_r.makespan
        out['fluent_savings'] = maximal_r.makespan - minimal_r.makespan
        return out
    finally:
        close_scene(C)


def sweep(seeds, run_timeout):
    """Each seed as its own subprocess with a wall-clock cap, same isolation demos/demo4.py's
    sweep() uses -- a crash or timeout only loses that seed (reported and skipped)."""
    runs = {}
    for seed in seeds:
        tmp = tempfile.mkdtemp(prefix='contribution3_')
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
        elif 'minimal_makespan' not in r:
            lines = [ln.strip() for ln in open(log) if ln.strip()]
            why = r.get('error') or (lines[-1] if lines else 'no output')
            r = {'status': f'failed: {why}'}
        else:
            r['status'] = 'ok'
        runs[seed] = r
        msg = (f"minimal={r['minimal_makespan']:.2f}  full={r['full_makespan']:.2f}  "
              f"least_commitment={r['least_commitment']}") if r['status'] == 'ok' else ''
        print(f"  seed={seed}: {r['status']} in {time.time() - start:.0f}s  {msg}", flush=True)

    ok = [r for r in runs.values() if r['status'] == 'ok']
    print(f"\n{len(ok)}/{len(seeds)} seeds succeeded.")
    if not ok:
        return

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return '-'
        return f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]" if len(vals) > 1 else f"{vals[0]:.3f}"

    print(f"  least-commitment: {sum(r['least_commitment'] for r in ok)}/{len(ok)} seeds")
    print(f"  full mutex makespan:   {stat([r['full_makespan'] for r in ok])}")
    print(f"  minimal makespan:      {stat([r['minimal_makespan'] for r in ok])}")
    print(f"  minimal vs full:       {stat([r['minimal_vs_full_pct'] for r in ok])}%")
    print(f"  same-motion async full mutex:    {stat([r['async_full_makespan'] for r in ok])}")
    print(f"  same-motion async minimal:       {stat([r['async_minimal_makespan'] for r in ok])}")
    print(f"  same-motion async savings:       {stat([r.get('async_savings') for r in ok])}")
    print(f"  fluent mode: Tamer chose the cheaper order: {sum(r['fluent_tamer_chose_cheaper'] for r in ok)}/{len(ok)} seeds")
    print(f"  fluent mode cheaper-order savings: {stat([r['fluent_savings'] for r in ok])}")


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
