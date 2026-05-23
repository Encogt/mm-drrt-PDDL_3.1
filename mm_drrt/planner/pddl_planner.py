"""
PDDL Planner Integration for MM-dRRT

This module provides the main orchestration for PDDL-based task planning in MM-dRRT.
It integrates the Unified Planning Framework to generate task plan skeletons.
"""

import sys
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus

get_environment().credits_stream = None

from mm_drrt.planner.pddl_problem_generator import generate_problem
from mm_drrt.utils.pddl_parser import parse_pddl_plan, _infer_from_fixed_obj


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
                 transit_duration=10, transfer_duration=10):
        """
        Initialize PDDL planner.

        Args:
            domain_file: Path to PDDL domain file (not used with UPF, kept for compatibility)
            planner_name: Name of planner to use (e.g., 'enhsp', 'tamer', 'pyperplan')
            timeout: Planning timeout in seconds
            transit_duration: Fixed duration (seconds) for transit (pick) actions
            transfer_duration: Fixed duration (seconds) for transfer (place) actions
        """
        self.domain_file = domain_file
        self.planner_name = planner_name
        self.timeout = timeout
        self.transit_duration = transit_duration
        self.transfer_duration = transfer_duration

    def generate_plan(self, env):
        """
        Generate task plan from environment.

        Args:
            env: MM-dRRT environment instance with create_pddl_problem() method

        Returns:
            tuple: (plan, action_orders, obj_orders, init_order_constraints)

        Raises:
            PDDLTimeoutError: If planning exceeds timeout
            PDDLUnsolvableError: If no valid plan exists
            PDDLParseError: If plan parsing fails
        """
        # Check that environment supports PDDL planning
        if not hasattr(env, 'create_pddl_problem'):
            raise NotImplementedError(
                f"Environment {type(env).__name__} must implement create_pddl_problem() method"
            )

        # Generate problem
        try:
            problem, mapper = generate_problem(env, save_to_file=True,
                                               transit_duration=self.transit_duration,
                                               transfer_duration=self.transfer_duration)
        except Exception as e:
            raise PDDLPlannerError(f"Problem generation failed: {e}")

        # Select planner — auto-select by default to ensure PDDL 3.1 support
        try:
            if self.planner_name is not None:
                with OneshotPlanner(name=self.planner_name) as planner:
                    result = planner.solve(problem, timeout=self.timeout)
            else:
                with OneshotPlanner(problem_kind=problem.kind) as planner:
                    result = planner.solve(problem, timeout=self.timeout)
        except Exception as e:
            if self.planner_name is not None:
                print(f"Warning: Planner '{self.planner_name}' failed: {e}")
                print("Falling back to auto-selection by problem kind...")
                try:
                    with OneshotPlanner(problem_kind=problem.kind) as planner:
                        result = planner.solve(problem, timeout=self.timeout)
                except Exception as e2:
                    raise PDDLPlannerError(f"Planning failed: {e2}")
            else:
                raise PDDLPlannerError(f"Planning failed: {e}")

        # Check result status
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            print(f"✓ PDDL planner found satisficing solution")
        elif result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
            print(f"✓ PDDL planner found optimal solution")
        elif result.status == PlanGenerationResultStatus.TIMEOUT:
            raise PDDLTimeoutError(f"Planning exceeded timeout of {self.timeout}s")
        elif result.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN:
            raise PDDLUnsolvableError("Problem proven unsolvable")
        elif result.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY:
            raise PDDLUnsolvableError("Problem likely unsolvable (incomplete check)")
        else:
            raise PDDLPlannerError(f"Planning failed with status: {result.status}")

        # Parse plan to MM-dRRT format
        try:
            pddl_plan = result.plan
            if pddl_plan is None:
                raise PDDLParseError("Planner returned None plan")

            timed = hasattr(pddl_plan, 'timed_actions')
            plan_len = len(pddl_plan.timed_actions) if timed else len(pddl_plan.actions)
            print(f"  Plan length: {plan_len} actions")

            # Parse to MM-dRRT format
            plan, action_orders, obj_orders, init_order_constraints = \
                parse_pddl_plan(pddl_plan, mapper, env)

            # Infer from_fixed_obj for transfer actions
            # We need to rebuild actions_list for this
            actions_list = []
            if hasattr(pddl_plan, 'timed_actions'):
                for i, (start_time, action, duration) in enumerate(pddl_plan.timed_actions):
                    action_name = f"a{i}"
                    from mm_drrt.utils.pddl_parser import _parse_action
                    action_info = _parse_action(action, action_name, mapper, start_time, start_time + duration)
                    actions_list.append(action_info)
            else:
                for i, action in enumerate(pddl_plan.actions):
                    action_name = f"a{i}"
                    from mm_drrt.utils.pddl_parser import _parse_action
                    action_info = _parse_action(action, action_name, mapper, i, i + 1)
                    actions_list.append(action_info)

            _infer_from_fixed_obj(actions_list)

            # Rebuild plan with updated from_fixed_obj
            plan = {action_info['name']: action_info['mm_drrt_format'] for action_info in actions_list}

        except Exception as e:
            raise PDDLParseError(f"Plan parsing failed: {e}")

        # Validate plan
        try:
            self._validate_plan(plan, action_orders, obj_orders, env)
        except Exception as e:
            print(f"Warning: Plan validation failed: {e}")
            print("Continuing anyway...")

        return plan, action_orders, obj_orders, init_order_constraints

    def _validate_plan(self, plan, action_orders, obj_orders, env):
        """
        Validate that plan is consistent with environment.

        Args:
            plan: MM-dRRT plan dict
            action_orders: Robot → action sequence mapping
            obj_orders: Object → action list mapping
            env: Environment instance

        Raises:
            ValueError: If plan is invalid
        """
        # Check that all actions in action_orders exist in plan
        for robot, actions in action_orders.items():
            for action_name in actions:
                if action_name not in plan:
                    raise ValueError(f"Action {action_name} in action_orders but not in plan")

        # Check that plan is not empty
        if len(plan) == 0:
            raise ValueError("Generated plan is empty")

        # Check that each robot (by PyBullet ID) has at least one action
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
    """
    Check if environment supports PDDL planning.

    Args:
        env: Environment instance

    Returns:
        bool: True if environment has create_pddl_problem() method
    """
    return hasattr(env, 'create_pddl_problem') and \
           callable(getattr(env, 'create_pddl_problem'))
