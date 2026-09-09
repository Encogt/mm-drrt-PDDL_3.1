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
  3. minimal constraint (region_mutex_enabled=True, transit_region_entry_offset=3,
                       transfer_region_exit_offset=7 -- the new encoding)

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
        stock = run(env, "1) stock (region_mutex_enabled=False, no region protection at all)")
        full = run(env, "2) region_mutex_enabled=True, default offsets (full-duration mutex)",
                  region_mutex_enabled=True)
        minimal = run(env, "3) region_mutex_enabled=True, entry=3/exit=7 (minimal constraint)",
                     region_mutex_enabled=True, transit_region_entry_offset=3,
                     transfer_region_exit_offset=7)
        print(f"Makespan -- stock: {stock:.2f}  full mutex: {full:.2f}  "
             f"minimal constraint: {minimal:.2f}")
        print("Minimal constraint sits between the two: safer than stock (no forced separation "
             "at all), faster than a full mutex (only the true occupied sub-interval is forced "
             "apart).")
    finally:
        disconnect(C)


if __name__ == '__main__':
    main()
