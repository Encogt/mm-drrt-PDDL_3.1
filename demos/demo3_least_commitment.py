#!/usr/bin/env python
"""Contribution 3: least-commitment conflict resolution that avoids unnecessary full-action
serialization.

Two arms each place a block on ONE shared drop_pad. With motion-derived timing (contribution 2),
the same problem is solved three ways:

  stock           no region constraint at all -- the two placements may occupy the pad at once;
  full mutex      the classic resource lock: the pad is held for each action's WHOLE duration, so
                  the two placements are fully serialized;
  minimal         the pad is held only during the measured occupancy interval [alpha, beta], so
                  Tamer only has to enforce  s_i + beta_ik <= s_j + alpha_jk  (or the reverse).

For the minimal schedule, every conflicting pair's solved start gap is compared against the
theoretical minimum from derive_minimal_constraint() -- the evidence that the solver commits to no
more separation than the geometry requires. Both constrained plans are refined and replayed. The
walkthrough ends with the per-(robot, region) fluent mode, where each robot keeps its own
interval and the cheaper of the two orders has to be chosen.

Usage:
    python demos/demo3_least_commitment.py            # GUI walkthrough
    python demos/demo3_least_commitment.py --no_gui   # headless
"""
import os
import random
import sys

import numpy as np

from _walkthrough import demo_args, Walkthrough, pipeline_opt, setup_scene, close_scene, gantt, \
    schedule_rows, replay_sync, table

from mm_drrt.pipeline_rai import refine, planner_settings
from mm_drrt.planner.tamer_pddl_planner import TamerPDDLPlanner, _solve_with_wall_clock_timeout
from mm_drrt.planner.pddl_problem_generator import generate_problem
from mm_drrt.utils.schedule_repair import check_least_commitment, is_least_commitment, offsets_by_action, \
    retime_schedule
from mm_drrt.utils.pddl_parser import parse_pddl_plan
from mm_drrt.utils.minimal_temporal_constraint import derive_minimal_constraint
from mm_drrt.utils.motion_timing import velocity_limits
from mm_drrt.utils.rai_motion_planner_utils import sample_region_offsets
from mm_drrt.utils import rai_utils as ru

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compare_region_fluents_demo import _TransferOnlyProblem  # noqa: E402


def solve(env, label, **kwargs):
    planner = TamerPDDLPlanner(timeout=60, **kwargs)
    plan, action_orders, obj_orders, constraints = planner.generate_plan(env)
    makespan = max(a['end'] for a in planner.last_schedule)
    return dict(label=label, plan=plan, obj_orders=obj_orders, constraints=constraints,
                schedule=planner.last_schedule, makespan=makespan)


def pad_overlap(schedule, measured):
    """How long the two transfers' MEASURED pad occupancy windows overlap in this schedule."""
    a, b = [x for x in schedule if x['type'] == 'transfer']
    alpha, beta = measured['transfer'][:2]
    lo = max(a['start'] + alpha, b['start'] + alpha)
    hi = min(a['start'] + beta, b['start'] + beta)
    return max(0.0, hi - lo)


def main():
    args = demo_args(__doc__.split('\n')[1])
    w = Walkthrough("Contribution 3 -- least-commitment conflict resolution", not args.no_gui)
    opt = pipeline_opt(w.use_gui, args.seed, '--env_type', 'exp_region_coordination_demo', '--num_robots', '2',
                       '--num_objs', '2', '--use_pddl_planner', '--region_mutex_enabled',
                       '--region_offsets_from_motion', '--durations_from_motion', '--region_offset_mode', 'two',
                       '--no_offsets_cache')
    C, env = setup_scene(opt)
    initial_world = env.save_world()
    try:
        w.step("Scenario",
               """Two arms, one shared drop_pad. Nothing links the two arms' tasks (different
               blocks, different zones), so only a constraint on the pad can keep the two
               placements from colliding there.""")

        w.step("Motion-derived timing (contribution 2)",
               """Durations and pad occupancy intervals are measured from representative
               trajectories (shortest of 3 samples per robot and region).""")
        measured = env.measure_region_timing()
        for t, (a, b, d) in measured.items():
            print(f"  {t:8s}: duration {d:.3f}s, occupies its region during [{a:.3f}, {b:.3f}]s")
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        durations = dict(transit_duration=measured['transit'][2], transfer_duration=measured['transfer'][2])
        w.pause()

        w.step("Solve three ways",
               """Same PDDL problem, same durations; only the region constraint differs.""")
        stock = solve(env, 'stock (no region constraint)', **durations)
        full = solve(env, 'full mutex (whole action)', region_mutex_enabled=True, **durations)
        kwargs, effective, _ = planner_settings(env, opt, measured, use_four=False)
        kwargs.pop('timeout')
        minimal = solve(env, 'minimal [alpha, beta]', **kwargs)
        occupancy = {t: measured[t][:2] for t in ('transit', 'transfer')}
        for s in (stock, full, minimal):
            print()
            occ = {a['name']: occupancy[a['type']] for a in s['schedule'] if a['type'] == 'transfer'}
            gantt(schedule_rows(env, s['schedule'], occupancy=occ),
                  title=f"  {s['label']}: makespan {s['makespan']:.3f}s  (# = measured pad occupancy)")
        print()
        overlap = pad_overlap(stock['schedule'], measured)
        print(f"  stock: the two placements' pad occupancy overlaps for {overlap:.3f}s"
              f"{' -- UNSAFE without dRRT* stepping in' if overlap > 0 else ''}")
        print(f"  full mutex makespan {full['makespan']:.3f}s vs minimal {minimal['makespan']:.3f}s "
              f"({100 * (1 - minimal['makespan'] / full['makespan']):.1f}% shorter)")
        w.pause()

        w.step("Is the minimal schedule least-commitment?",
               """For each conflicting pair: the gap Tamer actually left between the two starts,
               the gap the chosen order requires (beta_i - alpha_j), the theoretical minimum over
               both orders from derive_minimal_constraint(), and the gap full serialization would
               impose (the first action's whole duration).""")
        checks = check_least_commitment(minimal['schedule'], offsets_by_action(minimal['schedule'], effective))
        by_name = {a['name']: a for a in minimal['schedule']}
        table(['pair', 'region', 'actual gap', 'required', 'minimum', 'full serialization', 'order', 'verdict'],
              [(f"{c.first}->{c.second}", c.region, f"{c.actual_gap:.3f}", f"{c.required_gap:.3f}",
                f"{c.minimal_gap:.3f}", f"{by_name[c.first]['end'] - by_name[c.first]['start']:.3f}",
                'minimal' if c.minimal_order_chosen else 'NOT minimal', c.reason) for c in checks])
        print(f"\n  => {'least-commitment' if is_least_commitment(checks) else 'NOT least-commitment'}: "
              f"the remaining slack is Tamer's 0.01s epsilon separation.")
        w.pause()

        w.step("Refine both constrained plans",
               """Both go through the unchanged PlanSkeleton/dRRT* refinement and are replayed:
               first the fully serialized plan, then the minimal one, where the second arm starts
               its placement while the first is still retreating.""")
        for s in (full, minimal):
            env.restore_world(initial_world)
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            path, secs = refine(env, opt, s['plan'], s['obj_orders'], s['constraints'])
            print(f"  {s['label']}: refined in {secs:.1f}s, {len(path)} composite nodes")
            if w.use_gui:
                env.restore_world(initial_world)
                w.say(f"Replaying: {s['label']}")
                replay_sync(C, env, s['plan'], path, w.use_gui)
                w.pause()

        w.step("Per-(robot, region) intervals: choosing the cheaper order",
               """With numeric fluents each robot keeps its OWN measured interval for the pad, so
               the two orders cost different amounts: i-before-j needs beta_i - alpha_j,
               j-before-i needs beta_j - alpha_i. derive_minimal_constraint() says which is
               cheaper; Tamer's solved order is checked against it.""")
        env.restore_world(initial_world)
        per_robot = {}
        for r, key in ((0, 'r0'), (1, 'r1')):
            robot = env.robots[r]
            saved = env.save_world()
            out = sample_region_offsets(robot, env._arm, env._grasp_type, env.m_objs[r], env.f_objs[2], 'transfer',
                                        None, collision_objs=env.fixed_obstacles,
                                        velocity_limits=velocity_limits(ru._spec_of(robot)[0]))
            env.restore_world(saved)
            per_robot[key] = out
            print(f"  {key}: duration {out[2]:.3f}s, occupies drop_pad during [{out[0]:.3f}, {out[1]:.3f}]s")
        duration = max(per_robot['r0'][2], per_robot['r1'][2])
        region_offsets = {(k, 'drop_pad'): {'transfer': v[:2]} for k, v in per_robot.items()}
        problem, mapper = generate_problem(_TransferOnlyProblem(region_offsets), transit_duration=duration,
                                           transfer_duration=duration, region_mutex_enabled=True,
                                           use_region_fluents=True, region_offsets=region_offsets)
        status, solved = _solve_with_wall_clock_timeout(problem, 60)
        plan, _, _, _, schedule = parse_pddl_plan(solved, mapper, env, return_schedule=True)
        offsets = {a['name']: per_robot[a['robot']][:2] for a in schedule}
        # The split domain lets Tamer park a robot between its approach and occupy phases, which
        # stretches that action past its duration. Normalize: each action runs for `duration` and
        # starts alpha before its occupy phase does -- the wait moves to before the action.
        schedule = [dict(a, start=a['occupy_start'] - offsets[a['name']][0],
                         end=a['occupy_start'] - offsets[a['name']][0] + duration) for a in schedule]
        mc = derive_minimal_constraint(*per_robot['r0'][:2], *per_robot['r1'][:2])
        cheaper = 'r0' if mc.order == 'i_before_j' else 'r1'
        first = min(schedule, key=lambda a: a['occupy_start'])
        print(f"\n  r0 first needs a {mc.raw_gap_i_before_j:.3f}s gap, r1 first needs {mc.raw_gap_j_before_i:.3f}s "
              f"-> cheaper: {cheaper} first")
        gantt(schedule_rows(env, schedule), title=f"  Tamer's schedule ({first['robot']} first):")
        [c] = check_least_commitment(schedule, offsets)
        if c.minimal_order_chosen:
            print(f"  Tamer chose the cheaper order (gap {c.actual_gap:.3f}s vs minimum {c.minimal_gap:.3f}s).")
        else:
            w.say(f"""Tamer chose the MORE expensive order: it is a satisficing planner and does not
                  optimize makespan. The two placements share no robot or object, so their order
                  on the pad is a pure scheduling decision, and re-timing with
                  derive_minimal_constraint()'s order fixes it without touching the plan:""")
            durs = {a['name']: duration for a in schedule}
            solved_r = retime_schedule(schedule, offsets, durs)
            minimal_r = retime_schedule(schedule, offsets, durs, order='minimal')
            retimed = [dict(a, start=minimal_r.starts[a['name']], end=minimal_r.starts[a['name']] + durs[a['name']],
                            occupy_start=minimal_r.starts[a['name']] + offsets[a['name']][0],
                            occupy_end=minimal_r.starts[a['name']] + offsets[a['name']][1]) for a in schedule]
            gantt(schedule_rows(env, retimed), title=f"  re-timed with the minimal order ({cheaper} first):")
            print(f"  makespan {solved_r.makespan:.3f}s (Tamer's order) -> {minimal_r.makespan:.3f}s (minimal order)")
        w.pause()

        w.step("Summary",
               """The region constraint covers only the measured occupancy intervals, so the
               solver separates conflicting actions by exactly the theoretical minimum (up to
               Tamer's epsilon) instead of serializing whole actions -- a shorter makespan with the
               same safety guarantee.""")
    finally:
        close_scene(C)


if __name__ == '__main__':
    main()
