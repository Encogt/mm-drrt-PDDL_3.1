#!/usr/bin/env python
"""Contribution 4: asynchronous refinement and replanning, targeting reduced makespan and
multi-robot scalability.

N Franka arms (exp_round_table) stand around ONE shared drop_pad; each picks a block from its own
zone and places it on the pad, so contention grows with N. Tamer plans with motion-derived
intervals (contributions 2-3), the unchanged PlanSkeleton/dRRT* refines the plan, and then the SAME
refined paths are executed four ways:

  lock-step        synchronous: every robot waits for the slowest one at each dRRT* composite
                   node (how the composite path is played back);
  full mutex       asynchronous, but every pair of placements on the pad is fully serialized;
  minimal          asynchronous, re-timed so that only the measured pad-occupancy intervals of each
                   pair are kept apart (s_j - s_i >= beta_i - alpha_j);
  motion conflict  asynchronous, constrained only for pairs whose motions REALLY collide, over the
                   part of each action where they do.

Every asynchronous execution moves the robots relative to each other compared with what dRRT*
verified, so it is swept for robot-robot collisions on a 20 ms grid; a collision adds that pair's
conflict interval and the schedule is re-timed (the asynchronous replanning loop).

Usage:
    python demos/demo4_async_scaling.py                   # GUI walkthrough, N = 3
    python demos/demo4_async_scaling.py --num_robots 4    # GUI walkthrough, N = 4 (refinement ~1-2 min)
    python demos/demo4_async_scaling.py --sweep 2,3       # headless scaling table

Each N runs in under 5 minutes headless (N = 4 is the slow one: up to 2 dRRT* attempts of 60 s).
"""
import random
import time

import numpy as np

from _walkthrough import demo_args, Walkthrough, pipeline_opt, setup_scene, close_scene, gantt, \
    schedule_rows, timed_rows, lockstep_rows, replay_sync, replay_async, table

from mm_drrt.pipeline_rai import refine, planner_settings, handoff_constraints_only
from mm_drrt.planner.tamer_pddl_planner import TamerPDDLPlanner
from mm_drrt.utils.schedule_repair import check_least_commitment, is_least_commitment, offsets_by_action
from mm_drrt.utils.motion_timing import executed_action_timeline, lockstep_makespan, async_schedule, \
    detect_motion_conflicts


def extra_args(parser):
    parser.add_argument('--num_robots', type=int, default=3)
    parser.add_argument('--sweep', type=str, default=None, help='Comma-separated N values, e.g. 2,3,4 (headless)')
    parser.add_argument('--drrt_time_limit', type=int, default=60,
                        help='dRRT* time limit (s) per refinement attempt')
    parser.add_argument('--refine_attempts', type=int, default=2,
                        help='Refinement attempts (new random seed each) before giving up on this N')


def tamer(env, **kwargs):
    start = time.time()
    planner = TamerPDDLPlanner(timeout=120, **kwargs)
    plan, action_orders, obj_orders, constraints = planner.generate_plan(env)
    return dict(plan=plan, action_orders=action_orders, obj_orders=obj_orders, constraints=constraints,
                schedule=planner.last_schedule, secs=time.time() - start,
                makespan=max(a['end'] for a in planner.last_schedule))


def run(n, args, w):
    opt = pipeline_opt(w.use_gui, args.seed, '--env_type', 'exp_round_table', '--num_robots', str(n),
                       '--num_objs', str(n), '--use_pddl_planner', '--region_mutex_enabled',
                       '--region_offsets_from_motion', '--durations_from_motion', '--region_offset_mode', 'two',
                       '--drrt_order_constraints', 'handoff', '--no_offsets_cache',
                       '--drrt_time_limit', str(args.drrt_time_limit))
    C, env = setup_scene(opt)
    initial_world = env.save_world()
    out = {'N': n}
    try:
        w.step(f"N = {n} arms around one shared pad",
               """Each arm picks the block from its own zone (beside it, away from its neighbours) and
               places it on the central drop_pad. Every pair of placements contends for the pad.""")

        w.step("Motion-derived timing, then Tamer",
               """Durations and occupancy intervals are measured from representative trajectories.
               Tamer solves the problem twice: with a whole-action mutex on the pad (full
               serialization) and with only the measured [alpha, beta] intervals (minimal).""")
        measured = env.measure_region_timing()
        for t, (a, b, d) in measured.items():
            print(f"  {t:8s}: duration {d:.3f}s, occupancy [{a:.3f}, {b:.3f}]s")
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        full = tamer(env, region_mutex_enabled=True, transit_duration=measured['transit'][2],
                     transfer_duration=measured['transfer'][2])
        kwargs, effective, _ = planner_settings(env, opt, measured, use_four=False)
        kwargs.pop('timeout')
        minimal = tamer(env, **kwargs)
        out.update(tamer_full=full['secs'], tamer_min=minimal['secs'],
                   plan_full=full['makespan'], plan_min=minimal['makespan'])
        print()
        gantt(schedule_rows(env, full['schedule']), title=f"  Tamer, full mutex: makespan {full['makespan']:.3f}s")
        print()
        gantt(schedule_rows(env, minimal['schedule']), title=f"  Tamer, minimal: makespan {minimal['makespan']:.3f}s")
        checks = check_least_commitment(minimal['schedule'], offsets_by_action(minimal['schedule'], effective))
        print(f"\n  least-commitment check: {', '.join(f'{c.first}->{c.second} {c.reason}' for c in checks)}"
              f"  => {'least-commitment' if is_least_commitment(checks) else 'NOT least-commitment'}")
        w.pause()

        w.step("Refine (asynchronous MM-dRRT)",
               """The minimal plan is refined by the unchanged PlanSkeleton/dRRT*: each robot's
               subproblems advance independently in the composite search, and dRRT* guarantees the
               composite path is collision-free. Region timing is NOT pushed into dRRT* as
               whole-action precedence -- it is the schedule's job below.""")
        plan = minimal['plan']
        path, secs = None, 0.0
        for attempt in range(args.refine_attempts):
            # dRRT* has no backtracking: with several arms it can greedily drive into a dead end
            # (e.g. all arms at the pad at once, none able to retreat) and then only stops at its
            # time limit. Retry from scratch with a fresh random seed instead of hanging.
            env.restore_world(initial_world)
            random.seed(opt.seed + attempt)
            np.random.seed(opt.seed + attempt)
            try:
                path, t = refine(env, opt, plan, minimal['obj_orders'],
                                 handoff_constraints_only(plan, minimal['constraints']))
                secs += t
                break
            except SystemExit as e:
                secs += opt.drrt_time_limit
                print(f"  refinement attempt {attempt + 1} failed ({e}); retrying with a new seed")
        if path is None:
            raise SystemExit(f"no refinement within {args.refine_attempts} attempts of {opt.drrt_time_limit}s")
        out['refine'] = secs
        print(f"  refined in {secs:.1f}s; composite path has {len(path)} nodes")
        timeline = executed_action_timeline(env, path, minimal['action_orders'], plan)
        if len(timeline) != len(plan):
            print(f"  WARNING: only {len(timeline)}/{len(plan)} actions could be timed")
        w.pause()

        w.step("Execute the same paths four ways",
               """lock-step: robots synchronise at every composite node. The three asynchronous
               variants re-time Tamer's actions (same actions, same orders) with each action's
               MEASURED duration, differing only in what is kept apart on the pad; each is swept
               for collisions and re-timed on any hit.""")
        sync = lockstep_makespan(timeline)
        out['lockstep'] = sync
        gantt(lockstep_rows(timeline), title=f"  lock-step: makespan {sync:.3f}s")
        sched = minimal['schedule']
        pad_pairs = [(a['name'], b['name'], 0.0, timeline[a['name']]['duration'], 0.0, timeline[b['name']]['duration'])
                     for i, a in enumerate(sched) for b in sched[i + 1:]
                     if a['name'] in timeline and b['name'] in timeline and a['robot'] != b['robot']
                     and a['region'] == b['region']]
        motion = detect_motion_conflicts(env, timeline)
        motion_pairs = [(c.name_i, c.name_j, c.alpha_i, c.beta_i, c.alpha_j, c.beta_j) for c in motion]
        print(f"\n  motion-conflict sweep: {len(motion)} colliding pair(s) out of {len(pad_pairs)} pad pairs"
              + ''.join(f"\n    {c.name_i} [{c.alpha_i:.2f},{c.beta_i:.2f}] x {c.name_j} [{c.alpha_j:.2f},{c.beta_j:.2f}]"
                        for c in motion))
        results = {}
        for key, label, conflicts in (('async_full', 'full mutex', pad_pairs), ('async_min', 'minimal', None),
                                      ('async_motion', 'motion conflict', motion_pairs)):
            res = async_schedule(env, sched, timeline, conflicts=conflicts)
            results[key] = res
            out[key] = res.makespan if res.feasible else None
            print()
            if res.feasible:
                gantt(timed_rows(timeline, res.starts),
                      title=f"  async, {label}: makespan {res.makespan:.3f}s ({res.rounds} round(s), verified collision-free)")
            else:
                print(f"  async, {label}: no collision-free re-timing ({'; '.join(res.log)})")
            for line in res.log[:-1]:
                print(f"    {line}")
        w.pause()

        rows = [('lock-step (synchronous, dRRT* timing)', f"{sync:.3f}", '')]
        for key, label in (('async_full', 'async, full mutex on the pad'), ('async_min', 'async, minimal intervals'),
                           ('async_motion', 'async, motion conflicts only')):
            m = out[key]
            rows.append((label, f"{m:.3f}" if m else 'n/a', f"{100 * (m / sync - 1):+.1f}%" if m else ''))
        table(['execution', 'makespan (s)', 'change vs lock-step'], rows)
        w.pause()

        if w.use_gui:
            w.step("Replay: lock-step", "The composite path, all robots stepping through dRRT*'s nodes together.")
            replay_sync(C, env, plan, path, w.use_gui)
            best = min((k for k in ('async_motion', 'async_min') if results[k].feasible),
                       key=lambda k: results[k].makespan, default=None)
            if best:
                env.restore_world(initial_world)
                w.step("Replay: asynchronous",
                       f"""The same paths, each robot on its own clock ({'motion conflict' if best == 'async_motion'
                       else 'minimal'} re-timing). The terminal prints what every robot is doing.""")
                replay_async(C, env, plan, results[best].execution, w.use_gui, speed=args.speed)
                w.pause()
        return out
    finally:
        close_scene(C)


def main():
    args = demo_args(__doc__.split('\n')[1], extra_args)
    sweep = [int(x) for x in args.sweep.split(',')] if args.sweep else None
    if sweep:
        args.no_gui = True
    w = Walkthrough("Contribution 4 -- asynchronous refinement and replanning: makespan and scalability",
                    not args.no_gui)
    results = []
    for n in (sweep or [args.num_robots]):
        try:
            results.append(run(n, args, w))
        except SystemExit as e:  # dRRT* raises SystemExit on its time limit
            print(f"\n  N = {n}: {e}")
            results.append({'N': n})
    w.step("Scaling summary" if sweep else "Summary",
           """Makespans in seconds of velocity-limited motion. 'plan' columns are Tamer's schedules
           (with representative durations); the rest execute the same refined paths.""")
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else '-'
    table(['N', 'Tamer s (min)', 'refine s', 'plan full', 'plan min', 'lock-step', 'async full', 'async min',
           'async motion', 'best async vs lock-step'],
          [(r['N'], fmt(r.get('tamer_min')), fmt(r.get('refine')), fmt(r.get('plan_full')), fmt(r.get('plan_min')),
            fmt(r.get('lockstep')), fmt(r.get('async_full')), fmt(r.get('async_min')), fmt(r.get('async_motion')),
            (f"{100 * (min(v for v in (r.get('async_min'), r.get('async_motion')) if v) / r['lockstep'] - 1):+.1f}%"
             if r.get('lockstep') and (r.get('async_min') or r.get('async_motion')) else '-'))
           for r in results])


if __name__ == '__main__':
    main()
