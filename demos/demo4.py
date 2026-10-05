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
    python demos/demo4.py                   # GUI walkthrough, N = 3
    python demos/demo4.py --num_robots 4    # GUI walkthrough, N = 4 (refinement ~1-2 min)
    python demos/demo4.py --sweep 2,3,4                  # headless multi-seed table (seeds 0,1,2)
    python demos/demo4.py --sweep 5,6 --seeds 0          # up to 6 arms; failed runs are reported and skipped

Each N runs in under 5 minutes headless (N = 4 is the slow one: up to 2 dRRT* attempts of 60 s).
"""
import random
import sys
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
    parser.add_argument('--seeds', type=str, default='0,1,2', help='With --sweep: comma-separated seeds')
    parser.add_argument('--run_timeout', type=int, default=300, help='With --sweep: wall-clock cap (s) per (N, seed) run')
    parser.add_argument('--json', type=str, default=None, help='Write this run\'s results to a JSON file (used by --sweep)')
    parser.add_argument('--drrt_time_limit', type=int, default=60,
                        help='dRRT* time limit (s) per refinement attempt')
    parser.add_argument('--refine_attempts', type=int, default=2,
                        help='Refinement attempts (new random seed each) before giving up on this N')
    parser.add_argument('--path_attempts', type=int, default=3,
                        help='Refined paths tried (fresh seed each) until one gets a verified asynchronous '
                             'minimal schedule; the last one is kept if none does')


def region_overlaps(timeline, abs_occupancy):
    """[(action, action, seconds)] for every two different robots' occupancies of the SAME region
    that overlap in time. abs_occupancy: {name: (enter, leave)} in absolute time. Here only the
    placements on the shared pad can meet -- each pick is from the robot's own zone."""
    names = [n for n in abs_occupancy if abs_occupancy[n][0] is not None]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = timeline[a], timeline[b]
            if ta['robot_index'] == tb['robot_index'] or ta['region'] != tb['region']:
                continue
            d = min(abs_occupancy[a][1], abs_occupancy[b][1]) - max(abs_occupancy[a][0], abs_occupancy[b][0])
            if d > 0:
                out.append((a, b, d))
    return out


def occupancy_from_rows(rows):
    return {n: (os_, oe) for _, acts in rows for n, s, e, os_, oe in acts}


def summary_line(r):
    """One line: minimal vs the full-mutex baseline, and vs lock-step with both overlap counts."""
    full, mini, sync = r['async_full'], r['async_min'], r['lockstep']
    vs_full = 100 * (1 - mini / full)
    vs_sync = 100 * (mini / sync - 1)
    rel = f"within {abs(vs_sync):.1f}% of lock-step" if abs(vs_sync) < 5.0 else \
        f"{abs(vs_sync):.1f}% {'faster' if vs_sync < 0 else 'slower'} than lock-step"
    fb = ' (fell back to lock-step)' if r.get('async_min_fallback') else ''
    return (f"minimal{fb} is {abs(vs_full):.1f}% {'shorter' if vs_full >= 0 else 'longer'} than full mutex; it is "
            f"{rel} with {r['async_min_overlaps']} pad overlap(s), while lock-step had "
            f"{r['lockstep_overlaps']} overlap(s) ({r['lockstep_overlap_s']:.2f}s)")


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
        sched = minimal['schedule']
        path, timeline, min_res, secs = None, None, None, 0.0
        for k in range(args.path_attempts):
            # Best of K refined paths: dRRT*'s path decides whether the asynchronous re-timing can
            # be verified at all (e.g. an arm resting in another arm's placement path). Keep the
            # first path whose minimal-interval schedule is verified collision-free.
            path_k = None
            for attempt in range(args.refine_attempts):
                # dRRT* has no backtracking: with several arms it can greedily drive into a dead
                # end (e.g. all arms at the pad at once, none able to retreat) and then only stops
                # at its time limit. Retry from scratch with a fresh random seed instead of hanging.
                env.restore_world(initial_world)
                random.seed(opt.seed + 100 * k + attempt)
                np.random.seed(opt.seed + 100 * k + attempt)
                try:
                    path_k, t = refine(env, opt, plan, minimal['obj_orders'],
                                       handoff_constraints_only(plan, minimal['constraints']))
                    secs += t
                    break
                except SystemExit as e:
                    secs += opt.drrt_time_limit
                    print(f"  refinement attempt {attempt + 1} failed ({e}); retrying with a new seed")
            if path_k is None:
                continue
            timeline_k = executed_action_timeline(env, path_k, minimal['action_orders'], plan)
            res = async_schedule(env, sched, timeline_k)
            path, timeline, min_res = path_k, timeline_k, res
            out['path_attempts'] = k + 1
            if res.feasible:
                break
            if k + 1 < args.path_attempts:
                print(f"  path {k + 1}: no verified asynchronous minimal schedule ({res.log[-1].strip()}); "
                      f"refining again with a fresh seed")
        if path is None:
            raise SystemExit(f"no refinement within {args.refine_attempts} attempts of {opt.drrt_time_limit}s")
        out['refine'] = secs
        print(f"  refined in {secs:.1f}s ({out['path_attempts']} path(s) tried); composite path has {len(path)} nodes")
        if len(timeline) != len(plan):
            print(f"  WARNING: only {len(timeline)}/{len(plan)} actions could be timed")
        w.pause()

        w.step("Execute the same paths four ways",
               """lock-step: dRRT*'s own execution, robots synchronised at every composite node.
               dRRT* only avoids collisions -- it was given no pad mutex -- so two placements CAN
               overlap on the pad there (measured below). The three asynchronous variants re-time
               the actions (same actions, same orders) with each action's MEASURED duration and
               keep placements apart on the pad: whole actions (full mutex), only the measured
               occupancy intervals (minimal), or only where the motions really collide (motion
               conflict). Each is swept for collisions and re-timed on any hit.""")
        sync = lockstep_makespan(timeline)
        out['lockstep'] = sync
        gantt(lockstep_rows(timeline), title=f"  lock-step: makespan {sync:.3f}s")
        overlaps = region_overlaps(timeline, occupancy_from_rows(lockstep_rows(timeline)))
        print(f"  lock-step pad occupancy overlaps (mutex NOT honoured): "
              f"{', '.join(f'{a}/{b} {d:.2f}s' for a, b, d in overlaps) or 'none'}")
        out['lockstep_overlaps'] = len(overlaps)
        out['lockstep_overlap_s'] = sum(d for _, _, d in overlaps)
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
            res = min_res if key == 'async_min' else async_schedule(env, sched, timeline, conflicts=conflicts)
            results[key] = res
            # No verified asynchronous schedule -> execute the (always valid) lock-step path.
            out[key] = res.makespan if res.feasible else sync
            out[key + '_fallback'] = not res.feasible
            print()
            if res.feasible:
                gantt(timed_rows(timeline, res.starts),
                      title=f"  async, {label}: makespan {res.makespan:.3f}s ({res.rounds} round(s), verified collision-free)")
                ov = region_overlaps(timeline, occupancy_from_rows(timed_rows(timeline, res.starts)))
                print(f"    pad occupancy overlaps: {', '.join(f'{a}/{b} {d:.2f}s' for a, b, d in ov) or 'none'}")
            else:
                print(f"  async, {label}: no verified collision-free re-timing -> falls back to lock-step "
                      f"execution ({sync:.3f}s). Why: {res.log[-1].strip()}")
                out[key + '_why'] = res.log[-1].strip()
                ov = overlaps  # lock-step is what executes
            out[key + '_overlaps'] = len(ov)
            out[key + '_overlap_s'] = sum(d for _, _, d in ov)
            for line in res.log[:-1]:
                print(f"    {line}")
        w.pause()

        full_ms = out['async_full']
        rows = []
        for key, label in (('async_full', 'async, full mutex on the pad (baseline)'),
                           ('async_min', 'async, minimal intervals'),
                           ('async_motion', 'async, motion conflicts only'),
                           ('lockstep', 'lock-step (reference: no pad mutex)')):
            m = out[key]
            fell_back = out.get(key + '_fallback')
            rows.append((label + (' (fell back to lock-step)' if fell_back else ''), f"{m:.3f}",
                         '' if key == 'async_full' else f"{100 * (m / full_ms - 1):+.1f}%",
                         '' if key == 'lockstep' else f"{100 * (m / sync - 1):+.1f}%",
                         f"{out[key + '_overlaps']} ({out[key + '_overlap_s']:.2f}s)"))
        table(['execution', 'makespan (s)', 'vs full mutex', 'vs lock-step', 'pad overlaps'], rows)
        print('\n  ' + summary_line(out))
        w.pause()

        if w.use_gui:
            w.step("Replay: lock-step",
                   """The composite path, all robots stepping through dRRT*'s nodes together. This is
                   dRRT*'s collision-only execution: watch for two arms on the pad at once where the
                   overlap above says so -- the pad mutex is not part of it.""")
            replay_sync(C, env, plan, path, w.use_gui, minimal['action_orders'])
            # Replay the minimal-interval schedule: it is the one that honours the pad mutex
            # (motion-conflict timing may legitimately overlap placements that never collide).
            best = next((k for k in ('async_min', 'async_motion') if results[k].feasible), None)
            if best:
                env.restore_world(initial_world)
                w.step("Replay: asynchronous",
                       f"""The scene is reset to the start, then the same paths are replayed with each robot
                       on its own clock ({'minimal-interval' if best == 'async_min' else 'motion-conflict'}
                       re-timing){': placements never overlap their pad occupancy intervals' if best == 'async_min'
                       else ''}. The terminal prints what every robot is doing.""")
                replay_async(C, env, plan, results[best].execution, w.use_gui, speed=args.speed)
                w.pause()
        return out
    finally:
        close_scene(C)


def sweep(args, ns, seeds):
    """Every (N, seed) as its own headless subprocess with its own wall-clock cap, so a Tamer
    timeout, dRRT* failure or crash only loses that run (reported and skipped). Prints per-N
    mean [min, max] over the successful runs."""
    import json, os, subprocess, tempfile
    bad = [n for n in ns if not 2 <= n <= 6]
    if bad:
        raise SystemExit(f"--sweep supports N = 2..6 (the round-table scene); got {bad}")
    runs = {}
    for n in ns:
        for seed in seeds:
            tmp = tempfile.mkdtemp(prefix='demo4_')
            out, log = os.path.join(tmp, 'result.json'), os.path.join(tmp, 'run.log')
            cmd = [sys.executable, os.path.abspath(__file__), '--no_gui', '--num_robots', str(n), '--seed', str(seed),
                   '--drrt_time_limit', str(args.drrt_time_limit), '--refine_attempts', str(args.refine_attempts),
                   '--path_attempts', str(args.path_attempts), '--json', out]
            start = time.time()
            with open(log, 'w') as f:
                try:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=args.run_timeout)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
            r = json.load(open(out)) if os.path.exists(out) else {}
            if timed_out:
                r = {'status': f'timeout (> {args.run_timeout}s)'}
            elif 'lockstep' not in r:
                lines = [ln.strip() for ln in open(log) if ln.strip()]
                why = r.get('error') or next((ln for ln in reversed(lines) if 'Error' in ln or 'TIMEOUT' in ln),
                                             lines[-1] if lines else 'no output')
                r = {'status': f'failed: {why}'}
            else:
                r['status'] = 'ok'
            runs[(n, seed)] = r
            msg = (f"full {r['async_full']:.2f}  minimal {r['async_min']:.2f}{'*' if r.get('async_min_fallback') else ''}"
                   f"  lock-step {r['lockstep']:.2f} ({r['lockstep_overlaps']} pad overlaps)"
                   f"  paths tried {r.get('path_attempts', 1)}") if r['status'] == 'ok' else ''
            print(f"  N={n} seed={seed}: {r['status']} in {time.time() - start:.0f}s  {msg}", flush=True)

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return '-'
        return f"{np.mean(vals):.2f} [{min(vals):.2f}, {max(vals):.2f}]" if len(vals) > 1 else f"{vals[0]:.2f}"

    def ok_runs(n):
        return [runs[(n, s_)] for s_ in seeds if runs[(n, s_)]['status'] == 'ok']

    def overlaps(rs, key):
        return f"{sum(r[key + '_overlaps'] for r in rs)} ({sum(r[key + '_overlap_s'] for r in rs):.2f}s)"

    print("\nHeadline: avoid whole-action serialization while honouring the pad mutex"
          " -- async minimal vs async full mutex (same refined paths):")
    rows = []
    for n in ns:
        ok = ok_runs(n)
        fb = lambda k: sum(1 for r in ok if r.get(k + '_fallback'))
        gain = [100 * (r['async_min'] / r['async_full'] - 1) for r in ok]
        rows.append((n, f"{len(ok)}/{len(seeds)}",
                     f"{stat([r['async_full'] for r in ok])} ({fb('async_full')} fb)" if ok else '-',
                     f"{stat([r['async_min'] for r in ok])} ({fb('async_min')} fb)" if ok else '-',
                     f"{np.mean(gain):+.1f}%" if gain else '-',
                     overlaps(ok, 'async_full') if ok else '-', overlaps(ok, 'async_min') if ok else '-'))
    table(['N', 'runs', 'async full mutex', 'async minimal', 'minimal vs full', 'full: pad overlaps',
           'minimal: pad overlaps'], rows)

    print("\nReference: lock-step is dRRT*'s collision-only execution and is NOT given the pad mutex:")
    rows = []
    for n in ns:
        ok = ok_runs(n)
        fb = lambda k: sum(1 for r in ok if r.get(k + '_fallback'))
        vs_sync = [100 * (r['async_min'] / r['lockstep'] - 1) for r in ok if not r.get('async_min_fallback')]
        rows.append((n, f"{len(ok)}/{len(seeds)}", stat([r['lockstep'] for r in ok]),
                     f"{sum(1 for r in ok if r['lockstep_overlaps'])}/{len(ok)} runs, {overlaps(ok, 'lockstep')}"
                     if ok else '-',
                     f"{np.mean(vs_sync):+.1f}%" if vs_sync else '-',
                     f"{stat([r['async_motion'] for r in ok])} ({fb('async_motion')} fb)" if ok else '-',
                     f"{np.mean([r.get('path_attempts', 1) for r in ok]):.1f}" if ok else '-'))
    table(['N', 'runs', 'lock-step', 'lock-step pad overlaps', 'async minimal vs lock-step', 'async motion conflict',
           'paths tried'], rows)

    print("\nPlanned (Tamer, representative durations) -- full mutex vs minimal intervals:")
    rows = []
    for n in ns:
        ok = [r for r in ok_runs(n) if 'plan_full' in r]
        gain = [100 * (r['plan_min'] / r['plan_full'] - 1) for r in ok]
        rows.append((n, f"{len(ok)}/{len(seeds)}", stat([r['plan_full'] for r in ok]), stat([r['plan_min'] for r in ok]),
                     f"{np.mean(gain):+.1f}%" if gain else '-'))
    table(['N', 'runs', 'full mutex', 'minimal', 'minimal vs full'], rows)
    print("\n  runs = successful (N, seed) runs; failed or timed-out runs are listed above and skipped.")
    print("  fb   = fallback: no verified collision-free asynchronous schedule was found, so that run executes")
    print("         lock-step and counts with its lock-step makespan and lock-step pad overlaps.")
    print("  'async minimal vs lock-step' is the mean over runs without a minimal fallback.")


def main():
    args = demo_args(__doc__.split('\n')[1], extra_args)
    if args.sweep:
        sweep(args, [int(x) for x in args.sweep.split(',')], [int(x) for x in args.seeds.split(',')])
        return
    w = Walkthrough("Contribution 4 -- asynchronous refinement and replanning: makespan and scalability",
                    not args.no_gui)
    try:
        result = run(args.num_robots, args, w)
    except SystemExit as e:  # dRRT* raises SystemExit on its time limit
        print(f"\n  N = {args.num_robots}: {e}")
        result = {'N': args.num_robots, 'error': str(e)}
    except Exception as e:  # e.g. a Tamer timeout: report it instead of a traceback-only exit
        print(f"\n  N = {args.num_robots}: {type(e).__name__}: {e}")
        result = {'N': args.num_robots, 'error': f"{type(e).__name__}: {e}"}
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in result.items()}, f)
    w.step("Summary",
           """Makespans are seconds of velocity-limited motion, all on the same refined paths. The
           baseline is async full mutex (whole placements serialized on the pad); lock-step is a
           reference that does not honour the pad mutex at all.""")
    if 'lockstep' in result:
        print('  ' + summary_line(result))
    if result.get('async_min_fallback'):
        print("  fallback: no verified collision-free asynchronous schedule was found; the lock-step execution"
              " is used instead")


if __name__ == '__main__':
    main()
