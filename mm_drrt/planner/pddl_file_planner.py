"""Solves a HAND-EDITED PDDL 2.1 domain/problem file pair with Tamer and feeds the result into
the same execution pipeline as TamerPDDLPlanner (mm_drrt/planner/tamer_pddl_planner.py) -- the
difference is where the unified_planning Problem comes from: TamerPDDLPlanner always builds one
in-memory from env.create_pddl_problem(); this module parses real PDDL text via PDDLReader
instead, so editing that text (a :duration, a fact, a goal, or the syntax itself) directly
changes what gets solved and, if solvable, executed in the RAI sim.

Object identity: a hand-written problem file can't invent new physical robots/objects/surfaces --
whatever it references has to resolve to something that actually exists in the chosen
Environment. build_env_object_mapper() binds PDDL object names to real env objects by
convention: 'robot{i}' -> env.robots[i] (robots have no string identity of their own to match
against), while movable-obj/fixed-obj objects are matched by exact name equality against
env.m_objs/env.f_objs, since those already ARE the real RAI frame name strings (e.g. 'box0',
'table_start') -- see mm_drrt/pddl/problems/two_robots_relay_problem.pddl for a problem file
using exactly these names, for ExampleTwoRobotsRaiEnvironment.
"""
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

from mm_drrt.planner.pddl_problem_generator import ObjectMapper
from mm_drrt.planner.tamer_pddl_planner import (
    TamerPlannerError, TamerTimeoutError, TamerUnsolvableError, TamerParseError,
    UNSOLVABLE_STATUSES, SOLVED_STATUSES,
)
from mm_drrt.utils.pddl_parser import parse_pddl_plan
from unified_planning.engines import PlanGenerationResultStatus


def build_env_object_mapper(env):
    mapper = ObjectMapper()
    robots = env.robots.items() if isinstance(env.robots, dict) else enumerate(env.robots)
    for i, robot in robots:
        mapper.register(robot, f'robot{i}')
    for m_obj in env.m_objs:
        mapper.register(m_obj, m_obj)
    for f_obj in env.f_objs:
        mapper.register(f_obj, f_obj)
    return mapper


class PDDLFilePlannerError(TamerPlannerError):
    """Base exception for the load-from-PDDL-files path."""
    pass


def generate_plan_from_pddl_files(env, domain_path, problem_path, timeout=30):
    """Parses domain_path/problem_path as real PDDL 2.1 text, solves with Tamer, and returns the
    same (plan, action_orders, obj_orders, init_order_constraints) shape TamerPDDLPlanner.
    generate_plan() does -- ready to hand straight to PlanSkeleton.

    Raises PDDLFilePlannerError (or a TamerTimeoutError/TamerUnsolvableError subclass) with a
    message describing exactly what went wrong: a parse error in the text, an object name the
    environment doesn't have, an unsolvable/timed-out problem, or a plan-parsing failure --
    deliberately NOT falling back to any other planner, since the whole point of loading a
    hand-edited file is to see how the solver reacts to THIS text, not to paper over it.
    """
    try:
        problem = PDDLReader().parse_problem(domain_path, problem_path)
    except Exception as e:
        raise PDDLFilePlannerError(f"Failed to parse {domain_path} / {problem_path}: {e}")

    mapper = build_env_object_mapper(env)
    unmapped = [o.name for o in problem.all_objects if mapper.get_pybullet_obj(o.name) is None]
    if unmapped:
        valid = sorted(mapper.pddl_to_pybullet.keys())
        raise PDDLFilePlannerError(
            f"{problem_path} references object(s) {unmapped} that don't exist in "
            f"{type(env).__name__}. Valid object names for this environment: {valid}")

    try:
        with OneshotPlanner(name='tamer') as planner:
            result = planner.solve(problem, timeout=timeout)
    except Exception as e:
        raise PDDLFilePlannerError(f"Solving failed: {e}")

    if result.status in UNSOLVABLE_STATUSES:
        raise TamerUnsolvableError(f"Problem proven unsolvable: {result.status}")
    if result.status == PlanGenerationResultStatus.TIMEOUT:
        raise TamerTimeoutError(f"Planning exceeded timeout of {timeout}s")
    if result.status not in SOLVED_STATUSES:
        raise PDDLFilePlannerError(f"Tamer did not produce a plan. Status: {result.status}")

    num_actions = len(result.plan.timed_actions) if hasattr(result.plan, 'timed_actions') \
        else len(result.plan.actions)
    print(f"✓ Tamer found a plan with {num_actions} actions (from hand-edited PDDL files)")
    if hasattr(result.plan, 'timed_actions'):
        # The only place a :duration edit is actually visible downstream of Tamer -- nothing
        # past this point (parse_pddl_plan's extract_sequential_constraints, PlanSkeleton,
        # dRRT*) consumes these start/end times, so this print is the ONE way to confirm a
        # duration change reached the solver at all.
        for start_time, action, duration in sorted(result.plan.timed_actions, key=lambda t: t[0]):
            end_time = start_time + duration if duration is not None else start_time
            print(f"    [{float(start_time):>6.2f} -> {float(end_time):>6.2f}]  {action}")

    try:
        return parse_pddl_plan(result.plan, mapper, env)
    except Exception as e:
        raise TamerParseError(f"Plan parsing failed: {e}")
