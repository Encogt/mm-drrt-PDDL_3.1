#!/usr/bin/env python
"""Showcases the FULLY GENERAL per-(robot, region) numeric-fluent mode
(mm_drrt/planner/pddl_domain.py's use_region_fluents=True) -- the literal realization of the
paper's Oi = {(Rk, [alpha_ik, beta_ik])} -- driven by REAL motion-derived measurements from
RegionCoordinationDemoRaiEnvironment (examples/envs/region_coordination_demo_rai_env.py), with
BOTH sides of the window narrowed (unlike compare_region_constraints_rai_env.py, which is limited
to narrowing one side per action type -- see that script's docstring for why).

This uses a transfer-only problem (both robots already holding their object, goal is just to
place it on the shared drop_pad) rather than the full pick+place scenario: with the full scenario,
fully narrowing both sides of both transit AND transfer hangs Tamer's search well past 150s at
this environment's size (see docs/minimal_temporal_constraints.md) -- confirmed with plain float
constants too, so it isn't specific to numeric fluents. Dropping transit halves the action
templates (3 instead of 6) and this solves in well under a second. The tradeoff: a transfer-only
plan can't be handed to PlanSkeleton/dRRT* (mm_drrt/utils/rai_task_planner_utils.py's
individual_path_computation assumes each object goes through pick-then-place in that strict
order), so this script validates the fluent mechanism and the motion-derived measurements
end-to-end at the Tamer level -- not through the full motion-refinement pipeline. See
main_rai.py --region_offsets_from_motion for that (scalar-mode) validation instead.

What to look for in the output: robot0 and robot1 measure genuinely DIFFERENT (alpha, beta) for
the exact same region (drop_pad) -- the two arms' real approach geometry differs even in this
roughly symmetric layout -- and the solved schedule's occupy windows are packed back-to-back
(minimal separation) while robot0's depart phase overlaps robot1's occupy phase, something a
single global (alpha, beta) pair could never express.

Usage:
    python compare_region_fluents_demo.py
"""
import sys

import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner

up.shortcuts.get_environment().credits_stream = None

from mm_drrt.utils.rai_utils import connect, disconnect
from examples.envs.region_coordination_demo_rai_env import RegionCoordinationDemoRaiEnvironment
from mm_drrt.utils.rai_motion_planner_utils import sample_region_offsets
from mm_drrt.planner.pddl_problem_generator import generate_problem

DURATION = 10.0


class _TransferOnlyProblem:
    """robot0/robot1 already holding movable_obj0/movable_obj1; goal is to place both on the
    shared drop_pad. Isolates the region conflict from transit entirely -- see module docstring
    for why."""

    def __init__(self, region_offsets):
        self._region_offsets = region_offsets

    def create_pddl_problem(self):
        objects = {'robot': ['r0', 'r1'], 'movable-obj': ['m0', 'm1'], 'fixed-obj': ['drop_pad']}
        init_state = [
            ('holding', 'r0', 'm0'), ('holding', 'r1', 'm1'),
            ('obj-clear', 'm0'), ('obj-clear', 'm1'),
            ('surface-accessible', 'drop_pad'),
            ('robot-can-reach', 'r0', 'drop_pad'), ('robot-can-reach', 'r1', 'drop_pad'),
        ]
        goal_state = [
            ('obj-location', 'm0', 'drop_pad'), ('obj-location', 'm1', 'drop_pad'),
            ('robot-free', 'r0'), ('robot-free', 'r1'),
        ]
        return objects, init_state, goal_state


def measure_per_robot_offsets(env):
    """Real per-(robot, drop_pad) transfer offsets -- NOT collapsed across robots, unlike
    DurationConflictRaiEnvironment.measure_region_offsets()'s union -- since the whole point here
    is keeping them distinct."""
    offsets = {}
    for robot_key, robot_pddl in (('r0', 0), ('r1', 1)):
        result = sample_region_offsets(
            env.robots[robot_pddl], env._arm, env._grasp_type, env.m_objs[robot_pddl],
            env.f_objs[2], 'transfer', DURATION, collision_objs=env.fixed_obstacles)
        if result is None:
            raise RuntimeError(f"No valid grasp/IK/motion sample found for robot {robot_key}")
        offsets[robot_key] = result
    return offsets


def main():
    C = connect(use_gui=False)
    env = RegionCoordinationDemoRaiEnvironment(num_robots=2, num_objs=2, arm='left',
                                               grasp_type='top', sim_id=C, seed=0)
    try:
        measured = measure_per_robot_offsets(env)
        print(f"Measured (alpha, beta) per robot for drop_pad:")
        print(f"  r0: {measured['r0']}")
        print(f"  r1: {measured['r1']}")
        distinct = measured['r0'] != measured['r1']
        print(f"[{'PASS' if distinct else 'FAIL'}] the two robots' measured windows are distinct "
             f"(a single global constant couldn't express this)\n")

        region_offsets = {('r0', 'drop_pad'): {'transfer': measured['r0']},
                          ('r1', 'drop_pad'): {'transfer': measured['r1']}}
        problem, mapper = generate_problem(
            _TransferOnlyProblem(region_offsets), region_mutex_enabled=True,
            use_region_fluents=True, region_offsets=region_offsets)

        with OneshotPlanner(name='tamer') as planner:
            result = planner.solve(problem)
        print(f"Status: {result.status}")
        timed = sorted(result.plan.timed_actions, key=lambda t: t[0])
        for s, a, d in timed:
            print(f"    [{float(s):>6.2f} -> {float(s) + float(d):>6.2f}]  {a}")

        occupy = {str(a).split('(')[1].split(',')[0]: (float(s), float(s) + float(d))
                 for s, a, d in timed if 'transfer-occupy' in str(a)}
        (s0, e0), (s1, e1) = occupy.values()
        no_occupy_overlap = e0 <= s1 or e1 <= s0
        print(f"\n[{'PASS' if no_occupy_overlap else 'FAIL'}] the two robots' occupy phases "
             f"don't overlap (safety preserved)")

        ok = distinct and no_occupy_overlap
        print(f"\n{'PASSED' if ok else 'FAILED'}: fully general per-(robot, region) fluent mode "
             f"confirmed with real motion-derived data.")
        if not ok:
            sys.exit(1)
    finally:
        disconnect(C)


if __name__ == '__main__':
    main()
