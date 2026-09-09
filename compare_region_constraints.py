#!/usr/bin/env python
"""Solves the SAME problem against the OLD full-duration `occupied` mutex domain and the NEW
minimal region-temporal-constraint domain, side by side, so the effect is visible in one run
instead of two separate solve_pddl.py invocations.

OLD domain: mm_drrt/pddl/domains/mm_drrt_manipulation_full_mutex_baseline.pddl -- a frozen
snapshot of mm_drrt_manipulation.pddl from before region_entry_offset/region_exit_offset were
added (plain transit/transfer, `occupied` held for the action's whole duration). Not used by any
planner/env code path -- kept only for this comparison.

NEW domain: mm_drrt/pddl/domains/mm_drrt_manipulation.pddl -- transit/transfer split into
approach/occupy/depart phases so `occupied` is only held for the sub-interval that actually
touches the shared region (see that file's own comments, and mm_drrt/planner/pddl_domain.py's
docstring for the region_mutex_enabled/region_*_offset knobs on the Python-generated path).

Against the default problem (region_overlap_demo_problem.pddl -- two robots, different objects,
one shared surface, nothing else), this also ASSERTS the effect rather than just printing numbers
to eyeball: both domains must solve, the OLD plan must show zero cross-robot temporal overlap (a
full mutex leaves it no choice), the NEW plan must show at least one genuine cross-robot overlap
(the whole point of narrowing the mutex to a sub-interval), and NEW's makespan must be strictly
shorter. Exits non-zero if any of that fails. Passing a different --problem still prints the
comparison, but skips those asserts (a problem with unrelated, non-conflicting cross-robot actions
could legitimately overlap under the OLD domain too, so "zero overlap" isn't a general invariant).

Usage:
    python compare_region_constraints.py
    python compare_region_constraints.py --problem path/to/other_problem.pddl
"""
import argparse
import sys
import time

import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

up.shortcuts.get_environment().credits_stream = None

OLD_DOMAIN = 'mm_drrt/pddl/domains/mm_drrt_manipulation_full_mutex_baseline.pddl'
NEW_DOMAIN = 'mm_drrt/pddl/domains/mm_drrt_manipulation.pddl'
DEFAULT_PROBLEM = 'mm_drrt/pddl/problems/region_overlap_demo_problem.pddl'

UNSOLVABLE_STATUSES = (
    PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
)


def solve(domain_path, problem_path):
    problem = PDDLReader().parse_problem(domain_path, problem_path)
    start = time.time()
    with OneshotPlanner(name='tamer') as planner:
        result = planner.solve(problem)
    elapsed = time.time() - start

    if result.status in UNSOLVABLE_STATUSES or result.plan is None:
        return result.status, elapsed, None, None

    timed = sorted(result.plan.timed_actions, key=lambda t: t[0])
    makespan = max(float(s) + float(d) for s, d in ((t[0], t[2]) for t in timed))
    return result.status, elapsed, timed, makespan


def print_timeline(timed_actions):
    for start_time, action, duration in timed_actions:
        end_time = float(start_time) + float(duration)
        print(f"    [{float(start_time):>6.2f} -> {end_time:>6.2f}]  {action}")


def _robot_of(action):
    """The robot parameter is always parameters[0] -- see mm_drrt/utils/pddl_parser.py's
    _build_action_info, which relies on the same convention."""
    param = action.actual_parameters[0]
    return param.object().name if hasattr(param, 'object') else str(param)


def cross_robot_overlaps(timed_actions):
    """All (action_i, action_j) pairs from DIFFERENT robots whose [start, end] windows genuinely
    intersect. Empty under a full-duration mutex on the only shared resource; non-empty is exactly
    what a narrower, sub-interval-only mutex makes possible."""
    intervals = [(float(s), float(s) + float(d), _robot_of(a), a) for s, a, d in timed_actions]
    overlaps = []
    for i, (s1, e1, r1, a1) in enumerate(intervals):
        for s2, e2, r2, a2 in intervals[i + 1:]:
            if r1 != r2 and s1 < e2 and s2 < e1:
                overlaps.append((a1, a2))
    return overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problem', default=DEFAULT_PROBLEM)
    opt = parser.parse_args()

    print(f"Problem: {opt.problem}\n")

    print(f"=== OLD: full-duration mutex ({OLD_DOMAIN}) ===")
    old_status, old_elapsed, old_timed, old_makespan = solve(OLD_DOMAIN, opt.problem)
    print(f"Status: {old_status}  ({old_elapsed:.2f}s)")
    if old_timed is not None:
        print_timeline(old_timed)
        print(f"  Makespan: {old_makespan:.2f}")

    print(f"\n=== NEW: minimal region constraint ({NEW_DOMAIN}) ===")
    new_status, new_elapsed, new_timed, new_makespan = solve(NEW_DOMAIN, opt.problem)
    print(f"Status: {new_status}  ({new_elapsed:.2f}s)")
    if new_timed is not None:
        print_timeline(new_timed)
        print(f"  Makespan: {new_makespan:.2f}")

    if old_makespan is not None and new_makespan is not None:
        saved = old_makespan - new_makespan
        print(f"\nMakespan: {old_makespan:.2f} -> {new_makespan:.2f} "
             f"({'-' if saved >= 0 else '+'}{abs(saved):.2f}, "
             f"{'shorter' if saved > 0 else 'longer' if saved < 0 else 'unchanged'})")

    if opt.problem != DEFAULT_PROBLEM:
        print(f"\n(Skipping invariant asserts -- calibrated for the default {DEFAULT_PROBLEM}, "
             f"not an arbitrary --problem.)")
        return

    print("\n=== Invariant checks (calibrated for region_overlap_demo_problem.pddl) ===")
    checks = [
        ("OLD domain solved",
         old_status not in UNSOLVABLE_STATUSES and old_timed is not None),
        ("NEW domain solved",
         new_status not in UNSOLVABLE_STATUSES and new_timed is not None),
    ]
    old_overlaps = cross_robot_overlaps(old_timed) if old_timed is not None else None
    new_overlaps = cross_robot_overlaps(new_timed) if new_timed is not None else None
    if old_overlaps is not None:
        checks.append(("OLD plan has ZERO cross-robot overlap (full mutex leaves no choice)",
                       len(old_overlaps) == 0))
    if new_overlaps is not None:
        checks.append(("NEW plan has cross-robot overlap (partial overlap actually happened)",
                       len(new_overlaps) > 0))
    if old_makespan is not None and new_makespan is not None:
        checks.append((f"NEW makespan ({new_makespan:.2f}) < OLD makespan ({old_makespan:.2f})",
                       new_makespan < old_makespan))

    all_passed = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_passed = all_passed and passed

    if new_overlaps:
        print("\n  Overlapping action pairs in the NEW plan:")
        for a1, a2 in new_overlaps:
            print(f"    {a1}  <->  {a2}")

    if not all_passed:
        print("\nFAILED: minimal temporal constraints did not behave as expected on this problem.")
        sys.exit(1)
    print("\nPASSED: minimal region temporal constraints confirmed on this problem.")


if __name__ == '__main__':
    main()
