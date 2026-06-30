"""
Tamer Planner Integration for MM-dRRT (via ANML + Unified Planning Framework).

Generates ANML domain/problem files from the environment, reads them back with
UPF's ANMLReader, calls Tamer to solve the temporal problem, and converts the
resulting TimeTriggeredPlan to MM-dRRT format via the ANML parser.
"""

import tempfile
import unified_planning.shortcuts as up
from unified_planning.io import ANMLReader
from unified_planning.engines import PlanGenerationResultStatus

from mm_drrt.planner.anml_problem_generator import generate_anml_files
from mm_drrt.utils.anml_parser import parse_anml_upf_plan


class TamerPlannerError(Exception):
    pass


class TamerTimeoutError(TamerPlannerError):
    pass


class TamerUnsolvableError(TamerPlannerError):
    pass


class TamerParseError(TamerPlannerError):
    pass


class TamerPlanner:
    """
    Temporal planner using Tamer (via UPF + ANML) for MM-dRRT.

    Generates ANML domain/problem files, reads them with UPF's ANMLReader,
    solves with Tamer, and returns (plan, action_orders, obj_orders,
    init_order_constraints) in MM-dRRT format.
    """

    def __init__(self, timeout=30, transit_duration=10, transfer_duration=10):
        self.timeout = timeout
        self.transit_duration = transit_duration
        self.transfer_duration = transfer_duration
        up.get_environment().credits_stream = None

    def generate_plan(self, env):
        if not hasattr(env, 'create_pddl_problem'):
            raise NotImplementedError(
                f"Environment {type(env).__name__} must implement create_pddl_problem()"
            )

        # Write combined ANML file (domain + problem) to a temp directory
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                combined_path, mapper = generate_anml_files(
                    env, tmpdir,
                    transit_duration=self.transit_duration,
                    transfer_duration=self.transfer_duration,
                )

                # Read with UPF's ANML reader
                try:
                    reader = ANMLReader()
                    problem = reader.parse_problem(combined_path)
                except Exception as e:
                    raise TamerPlannerError(f"ANMLReader failed: {e}")

                # Solve with Tamer
                try:
                    with up.OneshotPlanner(name='tamer') as planner:
                        result = planner.solve(problem)
                except Exception as e:
                    raise TamerPlannerError(f"Tamer solve call failed: {e}")

                status = result.status
                if status == PlanGenerationResultStatus.TIMEOUT:
                    raise TamerTimeoutError(f"Tamer exceeded timeout of {self.timeout}s")
                if status in (PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
                              PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY):
                    raise TamerUnsolvableError("Problem proven unsolvable by Tamer")
                if result.plan is None:
                    raise TamerPlannerError(
                        f"Tamer did not return a plan (status: {status})\n"
                        f"Log: {result.log_messages}"
                    )

                print("Tamer found a plan (ANML)")

                # Parse the UPF plan back to MM-dRRT format using the ANML mapper
                try:
                    plan, action_orders, obj_orders, init_order_constraints = \
                        parse_anml_upf_plan(result.plan, mapper, env)
                except Exception as e:
                    raise TamerParseError(f"Plan parsing failed: {e}")

        except (TamerPlannerError, TamerTimeoutError, TamerUnsolvableError, TamerParseError):
            raise
        except Exception as e:
            raise TamerPlannerError(f"Unexpected error: {e}")

        self._validate_plan(plan, action_orders, env)
        return plan, action_orders, obj_orders, init_order_constraints

    def _validate_plan(self, plan, action_orders, env):
        if len(plan) == 0:
            raise ValueError("Generated plan is empty")
        for robot, actions in action_orders.items():
            for action_name in actions:
                if action_name not in plan:
                    raise ValueError(f"Action {action_name} in action_orders but not in plan")
        print(f"  Plan validation passed: {len(plan)} actions, {len(action_orders)} robots")


def has_anml_support(env):
    return hasattr(env, 'create_pddl_problem') and callable(getattr(env, 'create_pddl_problem'))
