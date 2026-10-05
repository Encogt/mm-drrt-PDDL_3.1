"""The RAI MR-TAMP pipeline main_rai.py runs, as importable functions (the demos under demos/
drive the same code with their own options):

  build_parser() / make_env()   CLI options and environment construction
  measure_planned()             motion-derived region offsets/durations before planning
  run_pipeline()                Tamer -> least-commitment check -> PlanSkeleton/dRRT* refinement ->
                                temporal repair (Steps 5a/5b) loop
  refine()                      PlanSkeleton/dRRT* refinement of one given plan
  replay_lockstep()             replays the composite path in the viewer
"""
import argparse
import random
import time
from types import SimpleNamespace

import numpy as np

from examples.envs.example_single_robot_rai_env import ExampleSingleRobotRaiEnvironment, \
    ExampleSingleRobotCameraSetup
from examples.envs.example_two_robots_rai_env import ExampleTwoRobotsRaiEnvironment, \
    ExampleTwoRobotsRaiCameraSetup
from examples.envs.stack_blocks_rai_env import StackBlocksRaiEnvironment, \
    StackBlocksRaiCameraSetup
from examples.envs.duration_conflict_rai_env import DurationConflictRaiEnvironment, \
    DurationConflictRaiCameraSetup
from examples.envs.region_coordination_demo_rai_env import RegionCoordinationDemoRaiEnvironment, \
    RegionCoordinationDemoCameraSetup
from examples.envs.round_table_rai_env import RoundTableRaiEnvironment, RoundTableCameraSetup
from mm_drrt.planner.rai_task_planner import PlanSkeleton
from mm_drrt.utils.rai_utils import set_camera_pose
from mm_drrt.utils.rai_motion_planner_utils import replay_composite_path
from mm_drrt.utils.temporal_repair import repair_region_offsets, DEFAULT_TOLERANCE
from mm_drrt.utils.region_offsets_cache import load_cached_offsets, save_cached_offsets
from mm_drrt.utils.schedule_repair import check_least_commitment, is_least_commitment, \
    offsets_by_action, durations_by_action, retime_schedule
from mm_drrt.utils.motion_timing import executed_action_timeline, collapse_by_type, detect_motion_conflicts, \
    contact_events


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_robots', type=int, default=1)
    parser.add_argument('--num_objs', type=int, default=1)
    parser.add_argument('--num_placement_samples', type=int, default=30)
    parser.add_argument('--num_base_samples', type=int, default=50)
    parser.add_argument('--num_arm_samples', type=int, default=20)
    parser.add_argument('--arm', type=str, default='left')
    parser.add_argument('--grasp_type', type=str, default='top')
    parser.add_argument('--env_type', type=str, default='exp_single_robot_rai')  # only option in the POC
    parser.add_argument('--use_gui', action='store_false')
    parser.add_argument('--use_debug', action='store_true')
    parser.add_argument('--drrt_num_iters', type=int, default=10)
    parser.add_argument('--drrt_time_limit', type=int, default=2000)
    # PDDL planner params
    parser.add_argument('--use_pddl_planner', action='store_true', help='Use automatic planning to generate task plan: tries Tamer first, falls back to Fast Downward -- both PDDL 2.1')
    parser.add_argument('--pddl_timeout', type=int, default=30, help='Timeout in seconds for each planner')
    parser.add_argument('--transit_duration', type=float, default=10, help='Declared PDDL :duration for transit actions (Tamer path) / naive_duration_executor wait unit')
    parser.add_argument('--transfer_duration', type=float, default=10, help='Declared PDDL :duration for transfer actions (Tamer path) / naive_duration_executor wait unit')
    # Minimal temporal constraints (Tamer path only, --use_pddl_planner): opt into the domain's
    # `occupied` region mutex and narrow it from an action's full duration down to [entry_offset,
    # exit_offset] -- the sub-interval during which it actually occupies the shared region -- so two
    # robots conflicting in that region only get forced apart across that sub-interval (si + beta_ik
    # <= sj + alpha_jk, or the reverse), preserving partial overlap between the rest of their
    # high-level actions whenever possible. Off by default: some environments (e.g.
    # DurationConflictRaiEnvironment) depend on the "stock" domain having no region-conflict handling
    # at all. See mm_drrt/planner/pddl_domain.py.
    parser.add_argument('--region_mutex_enabled', action='store_true', help='Enable the PDDL domain\'s occupied-region mutex (off by default; required for the --*_region_*_offset args below to have any effect)')
    parser.add_argument('--transit_region_entry_offset', type=float, default=0, help='Seconds into a transit action before it starts occupying its shared region (alpha for transit)')
    parser.add_argument('--transit_region_exit_offset', type=float, default=None, help='Seconds into a transit action when it stops occupying its shared region (beta for transit); defaults to --transit_duration')
    parser.add_argument('--transfer_region_entry_offset', type=float, default=0, help='Seconds into a transfer action before it starts occupying its shared region (alpha for transfer)')
    parser.add_argument('--transfer_region_exit_offset', type=float, default=None, help='Seconds into a transfer action when it stops occupying its shared region (beta for transfer); defaults to --transfer_duration')
    # Motion-derived offsets: replaces the --*_region_*_offset CLI constants above with real
    # alpha_ik/beta_ik measured from an actual IK/motion-planned trajectory (env.measure_region_offsets(),
    # e.g. examples/envs/duration_conflict_rai_env.py's -- only environments that implement it support
    # this flag). See that method's docstring for the asymmetric reduction it applies (only entry_offset
    # for transit, only exit_offset for transfer) and why: Tamer's search does not scale to narrowing
    # both sides of both action types at once at this environment's size -- confirmed empirically, not
    # a domain soundness issue. See docs/minimal_temporal_constraints.md.
    parser.add_argument('--region_offsets_from_motion', action='store_true', help='Replace the --*_region_*_offset constants with values measured from a real motion-planned trajectory (requires --region_mutex_enabled and an environment implementing measure_region_offsets())')
    parser.add_argument('--region_offset_mode', type=str, default='four', choices=['four', 'two'], help='With --region_offsets_from_motion: feed all four measured offsets (transit/transfer x entry/exit) to Tamer, falling back to the two-offset reduction (transit entry + transfer exit) if Tamer exceeds --pddl_timeout; "two" skips straight to the reduction')
    # C1: motion-derived durations. Times every measured trajectory under velocity-limited execution
    # (mm_drrt/utils/motion_timing.py) -- density-independent, unlike waypoint-index fractions -- and
    # uses the measured durations as the PDDL :duration of transit/transfer.
    parser.add_argument('--durations_from_motion', action='store_true', help='With --region_offsets_from_motion: also take transit/transfer durations from the measured motion, and measure alpha/beta in seconds of velocity-limited motion instead of index fractions of --*_duration')
    parser.add_argument('--joint_velocity_scale', type=float, default=1.0, help='Fraction of the joint velocity limits motion timing assumes execution runs at')
    # Temporal repair after refinement (docs/minimal_temporal_constraints.md, Step 5).
    parser.add_argument('--repair_strategy', type=str, default='retime', choices=['report', 'retime', 'resolve'], help='On a repair violation: report only (widen + cache for future runs); retime the existing plan without re-planning (Step 5a), falling back to resolve if infeasible; or resolve: widen offsets and re-run Tamer + refinement (Step 5b)')
    parser.add_argument('--max_resolve_attempts', type=int, default=2, help='Cap on Step 5b re-solves within one run')
    parser.add_argument('--repair_tolerance', type=float, default=DEFAULT_TOLERANCE, help='Absolute tolerance (s) before a measured offset/duration counts as a violation')
    parser.add_argument('--retime_max_makespan_factor', type=float, default=None, help='Step 5a deadline: a re-timing whose makespan exceeds this factor x the solved makespan is infeasible (falls back to re-solving). Unbounded by default')
    # C2: motion-based conflict detection after refinement.
    parser.add_argument('--detect_motion_conflicts', action='store_true', help='After refinement, sweep every pair of different robots\' executed actions for real robot-robot collisions; report unnecessary/unanticipated mutexes and constrain only colliding pairs when re-timing')
    parser.add_argument('--conflict_samples', type=int, default=0, help='Waypoints sampled per action in the motion conflict sweep (0 = every waypoint; subsampling can skip a short colliding window)')
    # Load a HAND-EDITED PDDL domain/problem file pair (mm_drrt/planner/pddl_file_planner.py) and
    # run whatever Tamer solves from it in this RAI env, instead of the normal
    # create_pddl_problem()-generated-in-memory path. No fallback to another planner on
    # failure/unsolvable -- the point is to see how the solver reacts to exactly this text.
    parser.add_argument('--pddl_domain_file', type=str, default=None, help='Path to a hand-edited PDDL domain file; defaults to mm_drrt/pddl/domains/mm_drrt_manipulation.pddl if --pddl_problem_file is set')
    parser.add_argument('--pddl_problem_file', type=str, default=None, help='Path to a hand-edited PDDL problem file, e.g. mm_drrt/pddl/problems/two_robots_relay_problem.pddl -- setting this loads the plan from these files instead of --use_pddl_planner\'s normal chain')
    # exp_duration_conflict_rai only: skip PlanSkeleton/dRRT*'s real inter-robot collision checking
    # and run mm_drrt/utils/naive_duration_executor.py instead, to demonstrate what a wrong
    # --transfer_duration causes when nothing else protects against it. See that module's docstring.
    # What PlanSkeleton/dRRT* is told about cross-robot ordering. 'all' (default): same-object
    # handoffs plus extract_shared_surface_constraints' whole-action precedence between
    # non-overlapping same-surface actions. 'handoff': handoffs only -- region timing is then left
    # to the schedule (Step 5a re-timing / asynchronous execution), not to dRRT*. dRRT*'s
    # is_violate_order_constraints is global (any robot waiting on an unmet precedence disables
    # connect_to_target for all), which with 4 arms stalls or slows the search for minutes.
    parser.add_argument('--drrt_order_constraints', type=str, default='all', choices=['all', 'handoff'], help='Cross-robot precedence passed to dRRT*: all (handoffs + whole-action same-surface precedence) or handoff only')
    parser.add_argument('--no_offsets_cache', action='store_true', help='Neither read nor write experiments/region_offsets_cache.json (independent runs, e.g. demos and seed sweeps)')
    parser.add_argument('--naive_playback', action='store_true', help='exp_duration_conflict_rai only: bypass dRRT* and play back on a naive, duration-trusting schedule')
    return parser


def make_env(opt, C):
    if opt.env_type == 'exp_single_robot_rai':
        set_camera_pose(C, camera_point=ExampleSingleRobotCameraSetup[0], target_point=ExampleSingleRobotCameraSetup[1])
        env = ExampleSingleRobotRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                               grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    elif opt.env_type == 'exp_two_robots_rai':
        set_camera_pose(C, camera_point=ExampleTwoRobotsRaiCameraSetup[0], target_point=ExampleTwoRobotsRaiCameraSetup[1])
        env = ExampleTwoRobotsRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                             grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    elif opt.env_type == 'exp_stack_blocks_rai':
        set_camera_pose(C, camera_point=StackBlocksRaiCameraSetup[0], target_point=StackBlocksRaiCameraSetup[1])
        env = StackBlocksRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                        grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    elif opt.env_type == 'exp_duration_conflict_rai':
        set_camera_pose(C, camera_point=DurationConflictRaiCameraSetup[0], target_point=DurationConflictRaiCameraSetup[1])
        env = DurationConflictRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                             grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    elif opt.env_type == 'exp_region_coordination_demo':
        set_camera_pose(C, camera_point=RegionCoordinationDemoCameraSetup[0], target_point=RegionCoordinationDemoCameraSetup[1])
        env = RegionCoordinationDemoRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                                   grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    elif opt.env_type == 'exp_round_table':
        set_camera_pose(C, camera_point=RoundTableCameraSetup[0], target_point=RoundTableCameraSetup[1])
        env = RoundTableRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                       grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
    else:
        raise ValueError('Unsupported env_type for the RAI POC: {}'.format(opt.env_type))
    return env


def widen(a, b):
    """Union of two (alpha, beta[, duration]) measurements -- min entry, max exit, max duration --
    the same "only ever widen" rule repair_region_offsets() uses. A duration is kept only if both
    sides have one."""
    out = (min(a[0], b[0]), max(a[1], b[1]))
    if len(a) > 2 and len(b) > 2:
        out += (max(a[2], b[2]),)
    elif len(a) > 2:
        out += (a[2],)
    return out


def measure_planned(env, opt):
    """Motion-derived planned offsets (merged with any cached widening), or None without
    --region_offsets_from_motion. Returns (planned_region_offsets, cache_key)."""
    if opt.durations_from_motion and not opt.region_offsets_from_motion:
        raise ValueError('--durations_from_motion requires --region_offsets_from_motion')

    # Planned region timing per action type: {'transit': (alpha, beta[, duration]), ...}. None unless
    # --region_offsets_from_motion. With --durations_from_motion every value is in seconds of
    # velocity-limited motion (mm_drrt/utils/motion_timing.py) and carries a 3rd element, the duration
    # itself; otherwise (alpha, beta) are waypoint fractions scaled to the declared --*_duration.
    planned_region_offsets = None
    # Cached values are only comparable within one unit system, so the motion-time mode gets its own key.
    # An env can scope its cache further (RoundTableRaiEnvironment: per number of arms).
    cache_key = getattr(env, 'cache_name', type(env).__name__) + (':motion_time' if opt.durations_from_motion else '')
    if opt.region_offsets_from_motion:
        if not opt.region_mutex_enabled:
            raise ValueError('--region_offsets_from_motion requires --region_mutex_enabled')
        if not hasattr(env, 'measure_region_offsets'):
            raise ValueError(f'{type(env).__name__} does not implement measure_region_offsets() -- '
                             f'--region_offsets_from_motion is not supported for --env_type {opt.env_type}')
        if opt.durations_from_motion:
            if not hasattr(env, 'measure_region_timing'):
                raise ValueError(f'{type(env).__name__} does not implement measure_region_timing() -- '
                                 f'--durations_from_motion is not supported for --env_type {opt.env_type}')
            print("Measuring region entry/exit times AND durations from real IK/motion-planned "
                  f"trajectories (velocity-limited, scale {opt.joint_velocity_scale})...")
            measured = env.measure_region_timing(vel_scale=opt.joint_velocity_scale)
        else:
            print("Measuring region entry/exit offsets from real IK/motion-planned trajectories...")
            measured = env.measure_region_offsets()
        print(f"  transit  = {measured['transit']}")
        print(f"  transfer = {measured['transfer']}")

        # A past run's repair widening (below) has nowhere to land if nothing ever reads it back --
        # merge any cached prior widening in now so a past violation's correction carries forward
        # instead of being silently re-measured away on the next run.
        cached = None if opt.no_offsets_cache else load_cached_offsets(cache_key)
        if cached:
            print(f"  cached from a prior run: {cached}")
            for action_type in ('transit', 'transfer'):
                if action_type in cached:
                    measured[action_type] = widen(measured[action_type], cached[action_type])
            print(f"  merged -> transit={measured['transit']}  transfer={measured['transfer']}")
        planned_region_offsets = measured

        # measure_region_offsets() draws random numbers (grasp shuffling, placement sampling) from the
        # same global random state everything else below uses -- without re-seeding, that shifts which
        # grasp/placement Steps 1-3 later draw purely as a side effect of THIS flag being on, making a
        # --region_offsets_from_motion run's chosen arm path differ from a plain run's even though path
        # computation itself (Steps 1-3) is completely unaffected by this feature. Re-seeding puts
        # everything after this point back on the exact same random draws either way, so the only real
        # difference between the two runs is the schedule this flag produces, not incidental path noise.
        random.seed(opt.seed)
        np.random.seed(opt.seed)
    return planned_region_offsets, cache_key


def planner_settings(env, opt, offsets, use_four):
    """Tamer durations/offsets for one solve. With measured offsets, all four (transit and transfer,
    entry and exit) are fed in when use_four; otherwise the older asymmetric reduction (transit
    entry + transfer exit only), which Tamer's search handles at this environment's size where the
    four-offset problem can run for minutes (see TamerPDDLPlanner's wall-clock guard).

    Returns (kwargs for TamerPDDLPlanner, effective {'transit': (alpha, beta), 'transfer': ...} --
    the occupancy window the domain really enforces with those kwargs)."""
    durations = {'transit': opt.transit_duration, 'transfer': opt.transfer_duration}
    if offsets is not None and opt.durations_from_motion:
        durations = {t: offsets[t][2] for t in durations}
    window = {
        'transit': [opt.transit_region_entry_offset, opt.transit_region_exit_offset],
        'transfer': [opt.transfer_region_entry_offset, opt.transfer_region_exit_offset],
    }
    if offsets is not None:
        window['transit'][0] = offsets['transit'][0]
        window['transfer'][1] = offsets['transfer'][1]
        if use_four:
            window['transit'][1] = offsets['transit'][1]
            window['transfer'][0] = offsets['transfer'][0]
        else:
            window['transit'][1] = None
            window['transfer'][0] = 0
    effective = {t: (window[t][0], durations[t] if window[t][1] is None else window[t][1])
                 for t in window}
    kwargs = dict(timeout=opt.pddl_timeout,
                  transit_duration=durations['transit'], transfer_duration=durations['transfer'],
                  region_mutex_enabled=opt.region_mutex_enabled,
                  transit_region_entry_offset=window['transit'][0],
                  transit_region_exit_offset=window['transit'][1],
                  transfer_region_entry_offset=window['transfer'][0],
                  transfer_region_exit_offset=window['transfer'][1])
    return kwargs, effective, durations


def solve_task_plan(env, opt, offsets):
    """Planner preference order when --use_pddl_planner is set: Tamer (solves the UPF problem
    directly) first, then classical Fast Downward (via MA-PDDL compilation) if Tamer fails,
    then the environment's manual plan as the last resort. Mirrors main.py.

    Returns (plan, action_orders, obj_orders, init_order_constraints, planner_used, schedule,
    effective, durations) -- schedule/effective/durations describe Tamer's solved timing and are
    None unless Tamer produced the plan."""
    if opt.pddl_problem_file:
        from mm_drrt.planner.pddl_file_planner import generate_plan_from_pddl_files, PDDLFilePlannerError
        from mm_drrt.planner.tamer_pddl_planner import TamerPlannerError
        from mm_drrt.planner.pddl_domain import get_domain_file_path

        domain_file = opt.pddl_domain_file or get_domain_file_path()
        print(f"Solving hand-edited PDDL files: domain={domain_file} problem={opt.pddl_problem_file}")
        try:
            result = generate_plan_from_pddl_files(env, domain_file, opt.pddl_problem_file,
                                                   timeout=opt.pddl_timeout)
        except (PDDLFilePlannerError, TamerPlannerError) as e:
            print(f"\n{type(e).__name__}: {e}\n\nNo fallback for --pddl_problem_file -- fix the "
                  f"PDDL text and re-run to see how the solver reacts.")
            raise SystemExit(1)
        return result + (f'Tamer (from PDDL files: {opt.pddl_problem_file})', None, None, None)

    if not opt.use_pddl_planner:
        return env.create_plan_order_constraints() + ('manual (create_plan_order_constraints)', None, None, None)

    from mm_drrt.planner.tamer_pddl_planner import TamerPDDLPlanner, TamerPlannerError, \
        TamerTimeoutError, has_tamer_pddl_support
    from mm_drrt.planner.pddl_planner import PDDLPlanner, PDDLPlannerError, PDDLTimeoutError, has_pddl_support

    if has_tamer_pddl_support(env):
        modes = [True, False] if offsets is not None and opt.region_offset_mode == 'four' else [False]
        for use_four in modes:
            kwargs, effective, durations = planner_settings(env, opt, offsets, use_four)
            label = 'all four region offsets' if use_four else 'the two-offset reduction'
            print(f"Trying Tamer planner (PDDL 2.1) to generate task plan"
                  f"{f' with {label}' if offsets is not None else ''}...")
            try:
                planner = TamerPDDLPlanner(**kwargs)
                result = planner.generate_plan(env)
                print(f"  durations={durations}  effective region windows={effective}")
                return result + ('Tamer (PDDL 2.1)', planner.last_schedule, effective, durations)
            except TamerTimeoutError as e:
                print(f"Tamer planning timed out with {label}: {e}")
                if use_four:
                    print("  Falling back to the two-offset reduction (transit entry + transfer exit).")
            except TamerPlannerError as e:
                print(f"Tamer planning failed: {e}")
                break
    else:
        print(f"Warning: Environment {opt.env_type} does not support Tamer PDDL planning")

    print("Falling back to classical Fast Downward planner...")
    if not has_pddl_support(env):
        print(f"Warning: Environment {opt.env_type} does not support PDDL planning")
    else:
        try:
            result = PDDLPlanner(timeout=opt.pddl_timeout).generate_plan(env)
            return result + ('Fast Downward (classical PDDL)', None, None, None)
        except PDDLTimeoutError as e:
            print(f"PDDL planner timeout: {e}")
        except PDDLPlannerError as e:
            print(f"PDDL planning failed: {e}")

    print("Falling back to manual plan specification...")
    return env.create_plan_order_constraints() + ('manual (create_plan_order_constraints)', None, None, None)


def print_least_commitment(schedule, effective):
    """C3 check: is Tamer's solved separation between conflicting actions the theoretical minimum
    derive_minimal_constraint() says the region requires?"""
    checks = check_least_commitment(schedule, offsets_by_action(schedule, effective))
    if not checks:
        print("Least-commitment check: no cross-robot pairs share a region.")
        return
    print("Least-commitment check (Tamer's start gaps vs. the theoretical minimum):")
    for c in checks:
        order = 'minimal order' if c.minimal_order_chosen else 'NON-minimal order'
        print(f"  {c.first} -> {c.second} on {c.region}: actual gap={c.actual_gap:.3f}  "
              f"required={c.required_gap:.3f}  minimal={c.minimal_gap:.3f}  ({order}, "
              f"slack={c.slack:+.3f}, occupancy slack={c.occupancy_slack:+.3f}) -> {c.reason}")
    verdict = "least-commitment" if is_least_commitment(checks) else "NOT least-commitment"
    print(f"  => schedule is {verdict} for every region pair")


def measure_executed(env, opt, composite_path, plan, action_orders, durations):
    """(executed per-type measurement in the planned units, executed per-action timeline or None).
    The timeline (mm_drrt/utils/motion_timing.py) is always built when the env can be timed; in the
    fraction mode the per-type values still come from env.measure_executed_region_offsets() so they
    stay comparable with measure_region_offsets()'s planned values."""
    timeline = None
    try:
        timeline = executed_action_timeline(env, composite_path, action_orders, plan,
                                            vel_scale=opt.joint_velocity_scale)
    except Exception as e:  # best-effort, like measure_executed_region_offsets
        print(f"Note: could not build the executed timeline: {type(e).__name__}: {e}")
    if opt.durations_from_motion:
        return (collapse_by_type(timeline) if timeline else {'transit': None, 'transfer': None}), timeline
    return env.measure_executed_region_offsets(composite_path), timeline


def repair_offsets(opt, planned, effective, durations, executed):
    """Step 5's comparison. A violation is judged against what Tamer's mutex windows (and
    durations) were actually scheduled around (`effective` -- e.g. under the two-offset reduction
    the transit window runs to the end of the action), but what gets STORED is the union of the
    measured planned offsets and the executed measurement, so a two-offset run never caches a
    full-duration window as if it had been measured. Returns (widened {'transit': (alpha, beta
    [, duration])}, any_violation)."""
    repaired, any_violation = {}, False
    for action_type in ('transit', 'transfer'):
        measured_exec = executed.get(action_type)
        if measured_exec is None:
            print(f"  {action_type}: nothing to repair against this run (no executed measurement)")
            continue
        pa, pb = effective[action_type]
        ma, mb = measured_exec[:2]
        result = repair_region_offsets(pa, pb, ma, mb, tolerance=opt.repair_tolerance)
        violated = result.violated
        detail = f"(entry_diff={result.entry_diff:+.3f}, exit_diff={result.exit_diff:+.3f})"
        if opt.durations_from_motion:
            pd, md = durations[action_type], measured_exec[2]
            # Taking longer than scheduled is the dangerous direction, same as exiting later.
            violated = violated or md - pd > opt.repair_tolerance
            detail += f" duration planned={pd:.3f} measured={md:.3f}"
        repaired[action_type] = widen(planned[action_type], measured_exec)
        print(f"  {action_type}: planned=({pa:.3f},{pb:.3f}) measured=({ma:.3f},{mb:.3f}) "
              f"-> {'VIOLATED' if violated else 'ok'}  {detail}")
        any_violation = any_violation or violated
    return repaired, any_violation


def print_motion_conflicts(env, conflicts, timeline, plan, action_orders):
    """C2 report: which shared-region pairs really collide, which don't (their mutex was
    unnecessary), and which collisions no region mutex anticipated."""
    robots = list(env.robots.values())
    colliding = {frozenset((c.name_i, c.name_j)) for c in conflicts}
    names = list(timeline)
    print("Motion-based conflict detection (pairwise robot-robot collision sweep):")
    for x, a in enumerate(names):
        for b in names[x + 1:]:
            ta, tb = timeline[a], timeline[b]
            if ta['robot_index'] == tb['robot_index']:
                continue
            shared = ta['region'] == tb['region']
            hit = frozenset((a, b)) in colliding
            if shared and not hit:
                print(f"  {a} / {b} share {ta['region']} but never collide -> mutex unnecessary for this pair")
            elif hit and not shared:
                print(f"  {a} ({ta['region']}) / {b} ({tb['region']}) COLLIDE with no shared region "
                      f"-> unanticipated conflict, no mutex covers it")
    for c in conflicts:
        print(f"  conflict {c.name_i} [{c.alpha_i:.3f},{c.beta_i:.3f}] x {c.name_j} "
              f"[{c.alpha_j:.3f},{c.beta_j:.3f}] ({c.hits} colliding waypoint pairs)")


def try_retime(opt, schedule, repaired, durations, timeline, conflicts):
    """Step 5a: re-time Tamer's actions under the widened offsets (or, with motion-detected
    conflicts, only the really-colliding pairs over their conflict intervals) without re-planning.
    Returns the RetimeResult if a feasible re-timing exists, else None."""
    if schedule is None:
        print("  Re-timing needs Tamer's solved schedule (this plan didn't come from Tamer).")
        return None
    if opt.durations_from_motion and timeline:
        durs = {n: timeline[n]['duration'] if n in timeline else d
                for n, d in durations_by_action(schedule).items()}
    else:
        durs = durations_by_action(schedule)
    conflict_tuples = None
    if conflicts is not None:
        # Conflict intervals are motion time; in the fraction mode schedule times are the declared
        # duration, so scale each action's interval onto its scheduled duration.
        def scale(name):
            if opt.durations_from_motion or not timeline[name]['duration']:
                return 1.0
            return durs[name] / timeline[name]['duration']
        conflict_tuples = [(c.name_i, c.name_j, c.alpha_i * scale(c.name_i), c.beta_i * scale(c.name_i),
                            c.alpha_j * scale(c.name_j), c.beta_j * scale(c.name_j)) for c in conflicts]
    original_makespan = max(a['end'] for a in schedule)
    max_makespan = original_makespan * opt.retime_max_makespan_factor \
        if opt.retime_max_makespan_factor else None
    result = retime_schedule(schedule, offsets_by_action(schedule, repaired), durs,
                             conflicts=conflict_tuples, max_makespan=max_makespan)
    if not result.feasible:
        print(f"  Re-timing INFEASIBLE (contradictory orders"
              f"{f' or makespan > {max_makespan:.3f}' if max_makespan else ''}).")
        return None
    print(f"  Re-timed schedule (same actions and order; makespan {result.original_makespan:.3f} "
          f"-> {result.makespan:.3f}):")
    for a in schedule:
        n = a['name']
        print(f"    {n} {a['type']:8s} start {a['start']:.3f} -> {result.starts[n]:.3f} "
              f"({result.deltas[n]:+.3f})  duration {durs[n]:.3f}")
    return result



def run_pipeline(env, opt, planned_region_offsets, cache_key):
    """Plan -> refine -> measure -> repair, re-solving up to --max_resolve_attempts. Returns a
    SimpleNamespace with the final attempt's plan, action_orders, schedule (Tamer's, or None),
    effective windows, durations, composite_path, executed timeline, motion conflicts, retimed
    (RetimeResult if Step 5a re-timed it), attempts and refinement_times."""
    assert opt.max_resolve_attempts >= 0
    initial_world = env.save_world()
    offsets = planned_region_offsets
    attempt = 0
    retimed = timeline = conflicts = None
    refinement_times = []
    while True:
        attempt += 1
        if attempt > 1:
            print(f"\n===== Re-solve attempt {attempt - 1}/{opt.max_resolve_attempts} with widened offsets "
                  f"{offsets} =====")
            env.restore_world(initial_world)
            random.seed(opt.seed)
            np.random.seed(opt.seed)

        timeline = conflicts = retimed = None
        plan, action_orders, obj_orders, init_order_constraints, planner_used, schedule, effective, durations = \
            solve_task_plan(env, opt, offsets)
        print(f"Planner used: {planner_used}")
        if schedule is not None and opt.region_mutex_enabled:
            print_least_commitment(schedule, effective)

        if opt.drrt_order_constraints == 'handoff':
            init_order_constraints = handoff_constraints_only(plan, init_order_constraints)

        assert opt.num_robots == len(action_orders), "Error: num_robots is not properly set"
        ps = PlanSkeleton(env, plan, obj_orders, init_order_constraints, opt.num_placement_samples, opt.use_debug)
        refinement_start = time.time()
        composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters, opt.drrt_time_limit)
        refinement_elapsed = time.time() - refinement_start
        refinement_times.append(refinement_elapsed)
        print(f"Plan refinement (Steps 1-4) took {refinement_elapsed:.2f}s wall-clock; "
              f"composite path has {len(composite_path)} waypoints")

        # Temporal repair (docs/minimal_temporal_constraints.md): compare what each region's mutex window
        # was PLANNED with against what the REAL, dRRT*-refined composite path measured for this run.
        if not opt.region_offsets_from_motion or effective is None:
            break
        if not hasattr(env, 'measure_executed_region_offsets'):
            print(f"Note: {type(env).__name__} does not implement measure_executed_region_offsets() "
                  f"-- skipping temporal repair check for this run.")
            break
        print("Measuring ACTUAL region entry/exit offsets from the executed composite path...")
        executed, timeline = measure_executed(env, opt, composite_path, plan, action_orders, durations)
        repaired, any_violation = repair_offsets(opt, offsets, effective, durations, executed)
        if repaired and not opt.no_offsets_cache:
            cache_path = save_cached_offsets(cache_key, repaired)
            print(f"  Saved widened offsets to {cache_path} for future runs.")

        conflicts = None
        if opt.detect_motion_conflicts and timeline:
            conflicts = detect_motion_conflicts(env, timeline, max_samples=opt.conflict_samples)
            print_motion_conflicts(env, conflicts, timeline, plan, action_orders)

        if not any_violation:
            break
        # dRRT*'s own inter_robots_collision_fn (rai_drrt_star.py) already independently verified THIS
        # run's composite path is collision-free, regardless of what the PDDL-level mutex window assumed
        # -- a violation means the schedule was built from offsets that under-covered the real geometry.
        print(f"TEMPORAL REPAIR VIOLATION: the planned mutex windows/durations did not cover this run's "
              f"real measured motion (tolerance {opt.repair_tolerance}s). dRRT*'s collision checking still "
              f"guarantees this path is collision-free; the SCHEDULE is what needs repair.")
        widened = {t: repaired.get(t, offsets[t]) for t in offsets}
        if opt.repair_strategy == 'report':
            print("  --repair_strategy report: widened offsets saved for future runs only.")
            break
        if opt.repair_strategy == 'retime':
            print("Step 5a: re-timing the existing plan under the measured offsets...")
            retimed = try_retime(opt, schedule, repaired, durations, timeline, conflicts)
            if retimed is not None:
                break
            print("  Falling back to Step 5b (re-plan with widened offsets).")
        if attempt > opt.max_resolve_attempts:
            print(f"WARNING: reached --max_resolve_attempts={opt.max_resolve_attempts}; keeping this "
                  f"attempt's plan. Widened offsets are cached for future runs.")
            break
        offsets = widened
    return SimpleNamespace(plan=plan, action_orders=action_orders, obj_orders=obj_orders,
                           init_order_constraints=init_order_constraints, planner_used=planner_used,
                           schedule=schedule, effective=effective, durations=durations,
                           composite_path=composite_path, timeline=timeline, conflicts=conflicts,
                           retimed=retimed, attempts=attempt, refinement_times=refinement_times,
                           offsets=offsets)


def release_targets_and_grippers(env, plan):
    """(release_targets, gripper_frames) for replaying `plan` -- see replay_composite_path."""
    robots = list(env.robots.values())
    # Keyed by (robot index, object name), not just object name -- an object can go through more
    # than one transfer action over the course of a plan (e.g. a relay handoff: one robot places
    # it at a mid-point, a second robot later places it at the final destination), and a plain
    # per-object dict can only hold one destination per object, so a later transfer action for
    # the same object silently overwrites an earlier robot's own destination. Confirmed via
    # instrumentation on the 2-robot relay: box0 has two transfer actions (a1: l_ arm -> mid, a3:
    # r_ arm -> end); a flat per-object dict left both robots using a3's 'end' target, so the
    # first robot's release snapped/reparented the object 0.87m away from where it actually
    # placed it.
    release_targets = {}
    for name, (a_type, a_robot, a_m_obj, a_from, a_to) in plan.items():
        if a_type == 'transfer':
            release_targets[(robots.index(a_robot), a_m_obj)] = a_to
    gripper_frames = [getattr(getattr(r, 'spec', None), 'gripper_frame', None) for r in robots]
    if all(g is None for g in gripper_frames):
        gripper_frames = None  # single-mobile-robot scenario: replay_composite_path's own default
    return release_targets, gripper_frames


def replay_lockstep(C, env, plan, composite_path, action_orders=None):
    """Replays composite_path in the viewer, all robots stepping together through every dRRT*
    composite node (the synchronous execution the composite path encodes). With action_orders,
    grasps/releases happen at the waypoint where the gripper really reaches the object/surface
    (motion_timing.contact_events) instead of at dRRT* node boundaries."""
    robots = list(env.robots.values())
    release_targets, gripper_frames = release_targets_and_grippers(env, plan)
    events = None
    if action_orders is not None:
        try:
            events = contact_events(executed_action_timeline(env, composite_path, action_orders, plan))
        except Exception as e:  # best effort: fall back to node-boundary grasps/releases
            print(f"Note: replaying with node-boundary grasps/releases ({type(e).__name__}: {e})")
    replay_composite_path(C, composite_path, env.get_joints(robots), release_targets,
                          gripper_frames=gripper_frames, events=events)


def refine(env, opt, plan, obj_orders, init_order_constraints):
    """PlanSkeleton/dRRT* refinement (Steps 1-4) of a given plan. Returns (composite_path,
    wall-clock seconds). dRRT* raises SystemExit on its own time limit -- callers that must survive
    that catch it."""
    ps = PlanSkeleton(env, plan, obj_orders, init_order_constraints, opt.num_placement_samples, opt.use_debug)
    start = time.time()
    composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters,
                                        opt.drrt_time_limit)
    return composite_path, time.time() - start


def handoff_constraints_only(plan, init_order_constraints):
    """--drrt_order_constraints handoff: keep only same-object handoffs."""
    return tuple(c for c in init_order_constraints if plan[c['pre']][2] == plan[c['post']][2])
