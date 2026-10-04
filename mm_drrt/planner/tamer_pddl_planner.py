"""
Tamer (UPF) PDDL 2.1 Planner Integration for MM-dRRT.

Solves the problem directly as a Unified Planning Framework Problem via the
Tamer engine, which natively supports the domain's durative actions (see
mm_drrt/planner/pddl_domain.py). This is tried first; main.py falls back to
classical Fast Downward planning (mm_drrt.planner.pddl_planner.PDDLPlanner)
if it fails.
"""

import multiprocessing as mp
from types import SimpleNamespace

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


def _solve_in_child(problem, conn):
    """Child-process body for _solve_with_wall_clock_timeout: sends back only plain data (status
    name + (start, action name, parameter names, duration) tuples) -- UPF plan objects hold
    references to the problem and aren't reliably picklable across processes."""
    try:
        with OneshotPlanner(name='tamer') as planner:
            result = planner.solve(problem)
        timed = None
        if result.status in SOLVED_STATUSES:
            timed = [(start, a.action.name, [str(p) for p in a.actual_parameters], duration)
                     for start, a, duration in result.plan.timed_actions]
        conn.send(('ok', result.status.name, timed))
    except Exception as e:
        conn.send(('error', f"{type(e).__name__}: {e}", None))
    finally:
        conn.close()


def _solve_with_wall_clock_timeout(problem, timeout):
    """Tamer ignores OneshotPlanner.solve()'s timeout (UserWarning: 'Tamer does not support
    timeout') and can search indefinitely -- e.g. with all four region offsets narrowed on
    DurationConflictRaiEnvironment, it did not return within 300s. So the solve runs in a forked
    child that is killed on expiry. Returns (status, plan) where plan mimics a TimeTriggeredPlan's
    `timed_actions` closely enough for parse_pddl_plan (action.action.name, actual_parameters)."""
    ctx = mp.get_context('fork')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_solve_in_child, args=(problem, child_conn), daemon=True)
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout):
            raise TamerTimeoutError(f"Planning exceeded wall-clock timeout of {timeout}s")
        kind, status_name, timed = parent_conn.recv()
    except EOFError:
        raise TamerPlannerError("Tamer child process exited without a result")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        parent_conn.close()

    if kind == 'error':
        raise TamerPlannerError(f"Planning failed: {status_name}")
    status = PlanGenerationResultStatus[status_name]
    plan = None
    if timed is not None:
        plan = SimpleNamespace(timed_actions=[
            (start, SimpleNamespace(action=SimpleNamespace(name=name), actual_parameters=params), duration)
            for start, name, params, duration in timed])
    return status, plan


class TamerPDDLPlanner:
    """
    Tamer-based PDDL 2.1 planner orchestrator for MM-dRRT.

    This class handles:
    1. Problem generation from environment (UPF Problem, via pddl_problem_generator)
    2. Solving with the Tamer engine
    3. Plan parsing to MM-dRRT format
    4. Error handling and validation
    """

    def __init__(self, timeout=30, transit_duration=10, transfer_duration=10,
                region_mutex_enabled=False,
                transit_region_entry_offset=0, transit_region_exit_offset=None,
                transfer_region_entry_offset=0, transfer_region_exit_offset=None):
        self.timeout = timeout
        self.transit_duration = transit_duration
        self.transfer_duration = transfer_duration
        self.region_mutex_enabled = region_mutex_enabled
        self.transit_region_entry_offset = transit_region_entry_offset
        self.transit_region_exit_offset = transit_region_exit_offset
        self.transfer_region_entry_offset = transfer_region_entry_offset
        self.transfer_region_exit_offset = transfer_region_exit_offset
        # The solved schedule (see pddl_parser._build_schedule) of the last successful
        # generate_plan(), for the least-commitment check / re-timing in schedule_repair.py.
        self.last_schedule = None

    def generate_plan(self, env):
        if not hasattr(env, 'create_pddl_problem'):
            raise NotImplementedError(
                f"Environment {type(env).__name__} must implement create_pddl_problem() method"
            )

        try:
            problem, mapper = generate_problem(
                env, transit_duration=self.transit_duration, transfer_duration=self.transfer_duration,
                region_mutex_enabled=self.region_mutex_enabled,
                transit_region_entry_offset=self.transit_region_entry_offset,
                transit_region_exit_offset=self.transit_region_exit_offset,
                transfer_region_entry_offset=self.transfer_region_entry_offset,
                transfer_region_exit_offset=self.transfer_region_exit_offset)
        except Exception as e:
            raise TamerPlannerError(f"Problem generation failed: {e}")

        status, solved_plan = _solve_with_wall_clock_timeout(problem, self.timeout)

        if status in UNSOLVABLE_STATUSES:
            raise TamerUnsolvableError(f"Problem proven unsolvable: {status}")
        if status == PlanGenerationResultStatus.TIMEOUT:
            raise TamerTimeoutError(f"Planning exceeded timeout of {self.timeout}s")
        if status not in SOLVED_STATUSES:
            raise TamerPlannerError(f"Tamer did not produce a plan. Status: {status}")

        print(f"✓ Tamer found a plan with {len(solved_plan.timed_actions)} actions")

        try:
            plan, action_orders, obj_orders, init_order_constraints, self.last_schedule = \
                parse_pddl_plan(solved_plan, mapper, env, return_schedule=True)
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
