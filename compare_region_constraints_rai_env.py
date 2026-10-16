#!/usr/bin/env python
"""Same showcase as compare_region_constraints.py, but through the REAL
DurationConflictRaiEnvironment (dual-Franka-arm scenario) and the actual TamerPDDLPlanner path
(mm_drrt/planner/pddl_problem_generator.py's generate_problem()), instead of a hand-written PDDL
problem file. Builds the RAI scene headlessly (no GUI window, no display needed) and solves the
SAME generated problem three ways:

  1. stock            (region_mutex_enabled=False, the historic default -- no region protection
                       at all; this is the exact scenario naive_duration_executor.py exists to
                       show is unsafe with a wrong --transfer_duration)
  2. full mutex        (region_mutex_enabled=True, default offsets -- `occupied` held for each
                       action's whole duration)
  3. minimal constraint (region_mutex_enabled=True, offsets MEASURED from a real IK/motion-planned
                       trajectory via DurationConflictRaiEnvironment.measure_region_offsets()
                       (examples/envs/duration_conflict_rai_env.py), which in turn uses
                       mm_drrt/utils/rai_motion_planner_utils.py's sample_region_offsets() --
                       instead of a hand-picked constant)

Why (3) still uses the SCALAR offset mode (use_region_fluents=False in
mm_drrt/planner/pddl_problem_generator.py's generate_problem()), not the fully general per-
(robot, region) numeric-fluent mode: that mode is real and tested (see
mm_drrt/planner/pddl_domain.py's use_region_fluents), but Tamer's search does not scale to this
environment's full size (2 robots x 2 objects x 3 surfaces) once BOTH transit and transfer are
fully split into all 3 phases (approach+occupy+depart) -- confirmed hanging past a 150s timeout
with numeric fluents, and REPRODUCED with plain float constants too (so this is a search/grounding
cost from having 6 action templates at this environment's branching factor, not something specific
to numeric fluents). What DOES solve in well under a second at this scale: only ONE side of each
action type's window narrowed away from its default (transit: only entry_offset set, exit stays at
the full duration; transfer: only exit_offset set, entry stays at 0) -- i.e. 2 phases per action
type, 4 templates total, matching this module's working "minimal constraint" case from before
measure_region_offsets() existed. measure_region_offsets() therefore reports the full measured
(alpha, beta) pair per action type (the true union across every region sampled), but only the
ONE side each action type can actually afford to spend at this scale is passed into the run(...)
call below -- conservative on the other side (never shrinks it), and still genuinely motion-derived
on the side that IS used, not hand-picked. See mm_drrt/planner/pddl_domain.py's docstring for what
this collapse gives up relative to the paper's per-region Oi, and
docs/minimal_temporal_constraints.md for the full writeup of this scaling limitation.

Usage:
    python compare_region_constraints_rai_env.py
"""
import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner

up.shortcuts.get_environment().credits_stream = None

from mm_drrt.utils.rai_utils import connect, disconnect
from examples.envs.duration_conflict_rai_env import DurationConflictRaiEnvironment
from mm_drrt.planner.pddl_problem_generator import generate_problem


def run(env, label, **kwargs):
    problem, mapper = generate_problem(env, **kwargs)
    with OneshotPlanner(name='tamer') as planner:
        result = planner.solve(problem)
    print(f"=== {label} ===  status: {result.status}")
    timed = sorted(result.plan.timed_actions, key=lambda t: t[0])
    for s, a, d in timed:
        print(f"    [{float(s):>6.2f} -> {float(s) + float(d):>6.2f}]  {a}")
    makespan = max(float(s) + float(d) for s, a, d in timed)
    print(f"  makespan: {makespan:.2f}\n")
    return makespan


def main():
    C = connect(use_gui=False)
    env = DurationConflictRaiEnvironment(num_robots=2, num_objs=2, arm='left', grasp_type='top',
                                         sim_id=C, seed=0)
    try:
        offsets = env.measure_region_offsets()
        print(f"Motion-derived offsets: transit alpha/beta = {offsets['transit']}, "
             f"transfer alpha/beta = {offsets['transfer']}\n")

        stock = run(env, "1) stock (region_mutex_enabled=False, no region protection at all)")
        full = run(env, "2) region_mutex_enabled=True, default offsets (full-duration mutex)",
                  region_mutex_enabled=True)
        # Only ONE side of each action type's window narrowed away from its default -- see this
        # module's docstring for why (Tamer's search doesn't scale to this environment's size once
        # BOTH sides of BOTH action types are narrowed at once). Still real, measured values; just
        # a conservative reduction of which of them get spent here.
        minimal = run(env, "3) region_mutex_enabled=True, motion-derived offsets (minimal constraint)",
                     region_mutex_enabled=True,
                     transit_region_entry_offset=offsets['transit'][0],
                     transfer_region_exit_offset=offsets['transfer'][1])
        print(f"Makespan -- stock: {stock:.2f}  full mutex: {full:.2f}  "
             f"minimal constraint: {minimal:.2f}")
        ok = stock <= minimal <= full
        print(f"[{'PASS' if ok else 'FAIL'}] minimal constraint's makespan lands between stock "
             f"and full-mutex baselines")
        if not ok:
            raise SystemExit(1)
    finally:
        disconnect(C)


if __name__ == '__main__':
    main()
