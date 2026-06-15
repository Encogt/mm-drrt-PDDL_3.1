"""
PDDL Planner Integration for MM-dRRT.

This module generates a classical PDDL instance, invokes Fast Downward, and
parses the resulting plan into MM-dRRT's abstract action format.
"""

import os
import tempfile

from mm_drrt.planner.classical_pddl import (
    generate_classical_problem_files,
    read_plan_file,
    run_fast_downward,
)
from mm_drrt.utils.pddl_parser import parse_pddl_plan


class PDDLPlannerError(Exception):
    """Base exception for PDDL planning failures"""
    pass


class PDDLTimeoutError(PDDLPlannerError):
    """Planner exceeded time limit"""
    pass


class PDDLUnsolvableError(PDDLPlannerError):
    """No valid plan exists"""
    pass


class PDDLParseError(PDDLPlannerError):
    """Error parsing PDDL plan"""
    pass


class PDDLPlanner:
    """
    Main PDDL planner orchestrator for MM-dRRT.

    This class handles:
    1. Problem generation from environment
    2. Planner invocation
    3. Plan parsing to MM-dRRT format
    4. Error handling and validation
    """

    def __init__(self, domain_file=None, planner_name=None, timeout=30,
                 transit_duration=10, transfer_duration=10,
                 fast_downward_search='astar(blind())'):
        self.domain_file = domain_file
        self.planner_name = planner_name
        self.timeout = timeout
        self.transit_duration = transit_duration
        self.transfer_duration = transfer_duration
        self.fast_downward_search = fast_downward_search

    def generate_plan(self, env):
        if not hasattr(env, 'create_pddl_problem'):
            raise NotImplementedError(
                f"Environment {type(env).__name__} must implement create_pddl_problem() method"
            )

        temp_dir = tempfile.mkdtemp(prefix='mm-drrt-pddl-')
        try:
            try:
                domain_path, problem_path, mapper = generate_classical_problem_files(env, temp_dir)
            except Exception as e:
                raise PDDLPlannerError(f"Problem generation failed: {e}")

            plan_path = os.path.join(temp_dir, 'sas_plan')

            try:
                result = run_fast_downward(
                    domain_path,
                    problem_path,
                    plan_path,
                    self.timeout,
                    search=self.fast_downward_search,
                    preferred_command=self.planner_name or self.domain_file,
                )
            except FileNotFoundError as e:
                raise PDDLPlannerError(str(e))
            except Exception as e:
                raise PDDLPlannerError(f"Planning failed: {e}")

            plan_lines = read_plan_file(plan_path)

            if not plan_lines:
                output = (result.stdout or '') + '\n' + (result.stderr or '')
                output_lower = output.lower()
                if 'time limit' in output_lower or 'timeout' in output_lower:
                    raise PDDLTimeoutError(f"Planning exceeded timeout of {self.timeout}s")
                if 'unsolvable' in output_lower or 'no solution' in output_lower:
                    raise PDDLUnsolvableError('Problem proven unsolvable')
                raise PDDLPlannerError(
                    f"Fast Downward did not produce a plan. Return code: {result.returncode}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            print(f"✓ Fast Downward found a plan with {len(plan_lines)} actions")

            try:
                plan, action_orders, obj_orders, init_order_constraints = parse_pddl_plan(
                    plan_lines,
                    mapper,
                    env,
                )
            except Exception as e:
                raise PDDLParseError(f"Plan parsing failed: {e}")

            try:
                self._validate_plan(plan, action_orders, obj_orders, env)
            except Exception as e:
                print(f"Warning: Plan validation failed: {e}")
                print("Continuing anyway...")

            return plan, action_orders, obj_orders, init_order_constraints
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _validate_plan(self, plan, action_orders, obj_orders, env):
        for robot, actions in action_orders.items():
            for action_name in actions:
                if action_name not in plan:
                    raise ValueError(f"Action {action_name} in action_orders but not in plan")

        if len(plan) == 0:
            raise ValueError("Generated plan is empty")

        if hasattr(env, 'robots') and len(env.robots) > 0:
            robot_ids = set(env.robots.values()) if isinstance(env.robots, dict) else set(env.robots)
            for robot_id in robot_ids:
                if robot_id not in action_orders:
                    print(f"Warning: Robot {robot_id} has no actions in plan")

        print(f"  Plan validation passed:")
        print(f"    - {len(plan)} total actions")
        print(f"    - {len(action_orders)} robots")
        print(f"    - {len(obj_orders)} movable objects")


def has_pddl_support(env):
    return hasattr(env, 'create_pddl_problem') and callable(getattr(env, 'create_pddl_problem'))
