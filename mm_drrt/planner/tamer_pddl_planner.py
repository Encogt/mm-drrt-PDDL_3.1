"""
Tamer (UPF) PDDL 2.1 Planner Integration for MM-dRRT.

Solves the problem directly as a Unified Planning Framework Problem via the
Tamer engine, which natively supports the domain's durative actions (see
mm_drrt/planner/pddl_domain.py). This is tried first; main.py falls back to
classical Fast Downward planning (mm_drrt.planner.pddl_planner.PDDLPlanner)
if it fails.
"""

import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

from mm_drrt.planner.pddl_problem_generator import generate_problem
from mm_drrt.utils.pddl_parser import parse_pddl_plan

up.shortcuts.get_environment().credits_stream = None

UNSOLVABLE_STATUSES = (
    PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
)
SOLVED_STATUSES = (
    PlanGenerationResultStatus.SOLVED_SATISFICING,
    PlanGenerationResultStatus.SOLVED_OPTIMALLY,
)


class TamerPlannerError(Exception):
    """Base exception for Tamer planning failures"""
    pass


class TamerTimeoutError(TamerPlannerError):
    """Planner exceeded time limit"""
    pass


class TamerUnsolvableError(TamerPlannerError):
    """No valid plan exists"""
    pass


class TamerParseError(TamerPlannerError):
    """Error parsing Tamer's plan"""
    pass


class TamerPDDLPlanner:
    """
    Tamer-based PDDL 2.1 planner orchestrator for MM-dRRT.

    This class handles:
    1. Problem generation from environment (UPF Problem, via pddl_problem_generator)
    2. Solving with the Tamer engine
    3. Plan parsing to MM-dRRT format
    4. Error handling and validation
    """

    def __init__(self, timeout=30, transit_duration=10, transfer_duration=10):
        self.timeout = timeout
        self.transit_duration = transit_duration
        self.transfer_duration = transfer_duration

    def generate_plan(self, env):
        if not hasattr(env, 'create_pddl_problem'):
            raise NotImplementedError(
                f"Environment {type(env).__name__} must implement create_pddl_problem() method"
            )

        try:
            problem, mapper = generate_problem(env, transit_duration=self.transit_duration,
                                               transfer_duration=self.transfer_duration)
        except Exception as e:
            raise TamerPlannerError(f"Problem generation failed: {e}")

        try:
            with OneshotPlanner(name='tamer') as planner:
                result = planner.solve(problem, timeout=self.timeout)
        except Exception as e:
            raise TamerPlannerError(f"Planning failed: {e}")

        if result.status in UNSOLVABLE_STATUSES:
            raise TamerUnsolvableError(f"Problem proven unsolvable: {result.status}")
        if result.status == PlanGenerationResultStatus.TIMEOUT:
            raise TamerTimeoutError(f"Planning exceeded timeout of {self.timeout}s")
        if result.status not in SOLVED_STATUSES:
            raise TamerPlannerError(f"Tamer did not produce a plan. Status: {result.status}")

        num_actions = len(result.plan.timed_actions) if hasattr(result.plan, 'timed_actions') \
            else len(result.plan.actions)
        print(f"✓ Tamer found a plan with {num_actions} actions")

        try:
            plan, action_orders, obj_orders, init_order_constraints = parse_pddl_plan(
                result.plan, mapper, env,
            )
        except Exception as e:
            raise TamerParseError(f"Plan parsing failed: {e}")

        try:
            self._validate_plan(plan, action_orders, obj_orders, env)
        except Exception as e:
            print(f"Warning: Plan validation failed: {e}")
            print("Continuing anyway...")

        return plan, action_orders, obj_orders, init_order_constraints

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


def has_tamer_pddl_support(env):
    return hasattr(env, 'create_pddl_problem') and callable(getattr(env, 'create_pddl_problem'))
