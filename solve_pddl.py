#!/usr/bin/env python
"""Standalone PDDL 2.1 sandbox, decoupled from the RAI simulation entirely: parse a domain +
problem pair as plain text and solve with Tamer, printing exactly what the solver did with it.

Meant for hand-editing either file and re-running to see how the solver reacts -- a wrong
:duration, a missing precondition fact, an unreachable goal, broken syntax, etc. -- without
needing to touch main_rai.py, an Environment class, or any robot/motion-planning code at all.

Usage:
    python solve_pddl.py
    python solve_pddl.py --problem path/to/other_problem.pddl
    python solve_pddl.py --domain path/to/other_domain.pddl --problem path/to/other_problem.pddl
"""
import argparse
import time

import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.exceptions import UPUsageError, UPProblemDefinitionError

up.shortcuts.get_environment().credits_stream = None

DEFAULT_DOMAIN = 'mm_drrt/pddl/domains/mm_drrt_manipulation.pddl'
DEFAULT_PROBLEM = 'mm_drrt/pddl/problems/two_robots_relay_problem.pddl'

UNSOLVABLE_STATUSES = (
    PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', default=DEFAULT_DOMAIN)
    parser.add_argument('--problem', default=DEFAULT_PROBLEM)
    opt = parser.parse_args()

    print(f"Domain:  {opt.domain}")
    print(f"Problem: {opt.problem}")

    try:
        problem = PDDLReader().parse_problem(opt.domain, opt.problem)
    except (UPUsageError, UPProblemDefinitionError, SyntaxError) as e:
        print(f"\nPARSE ERROR -- the domain/problem text isn't valid PDDL 2.1:\n  {e}")
        return
    except Exception as e:
        print(f"\nPARSE ERROR ({type(e).__name__}):\n  {e}")
        return

    start = time.time()
    try:
        with OneshotPlanner(name='tamer') as planner:
            result = planner.solve(problem)
    except Exception as e:
        print(f"\nSOLVER ERROR ({type(e).__name__}):\n  {e}")
        return
    elapsed = time.time() - start

    print(f"\nStatus: {result.status}  ({elapsed:.2f}s)")
    if result.status in UNSOLVABLE_STATUSES:
        print("No plan exists for this problem under this domain -- the goal is provably "
             "unreachable given the current facts/preconditions/durations.")
    elif result.status == PlanGenerationResultStatus.TIMEOUT:
        print("Tamer timed out before finding or ruling out a plan.")
    elif result.plan is None:
        print("No plan returned (and not reported unsolvable) -- see status above.")
    else:
        print("Plan:")
        if hasattr(result.plan, 'timed_actions'):
            for start_time, action, duration in sorted(result.plan.timed_actions, key=lambda t: t[0]):
                end_time = start_time + duration if duration is not None else start_time
                print(f"  [{float(start_time):>6.2f} -> {float(end_time):>6.2f}]  {action}")
        else:
            for i, action in enumerate(result.plan.actions):
                print(f"  {i}: {action}")


if __name__ == '__main__':
    main()
