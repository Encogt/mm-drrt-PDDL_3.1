import argparse
import random
import time
import numpy as np

from mm_drrt.utils.rai_utils import connect, disconnect, set_camera_pose, refresh_view

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
from mm_drrt.planner.rai_task_planner import PlanSkeleton
from mm_drrt.utils.rai_motion_planner_utils import replay_composite_path
from mm_drrt.utils.temporal_repair import repair_region_offsets, DEFAULT_TOLERANCE
from mm_drrt.utils.region_offsets_cache import load_cached_offsets, save_cached_offsets
from experiments.data_saver import data_saver

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
# Load a HAND-EDITED PDDL domain/problem file pair (mm_drrt/planner/pddl_file_planner.py) and
# run whatever Tamer solves from it in this RAI env, instead of the normal
# create_pddl_problem()-generated-in-memory path. No fallback to another planner on
# failure/unsolvable -- the point is to see how the solver reacts to exactly this text.
parser.add_argument('--pddl_domain_file', type=str, default=None, help='Path to a hand-edited PDDL domain file; defaults to mm_drrt/pddl/domains/mm_drrt_manipulation.pddl if --pddl_problem_file is set')
parser.add_argument('--pddl_problem_file', type=str, default=None, help='Path to a hand-edited PDDL problem file, e.g. mm_drrt/pddl/problems/two_robots_relay_problem.pddl -- setting this loads the plan from these files instead of --use_pddl_planner\'s normal chain')
# exp_duration_conflict_rai only: skip PlanSkeleton/dRRT*'s real inter-robot collision checking
# and run mm_drrt/utils/naive_duration_executor.py instead, to demonstrate what a wrong
# --transfer_duration causes when nothing else protects against it. See that module's docstring.
parser.add_argument('--naive_playback', action='store_true', help='exp_duration_conflict_rai only: bypass dRRT* and play back on a naive, duration-trusting schedule')

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

C = connect(use_gui=opt.use_gui)
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
else:
    raise ValueError('Unsupported env_type for the RAI POC: {}'.format(opt.env_type))

refresh_view(C, use_gui=opt.use_gui)

if opt.region_offsets_from_motion:
    if not opt.region_mutex_enabled:
        raise ValueError('--region_offsets_from_motion requires --region_mutex_enabled')
    if not hasattr(env, 'measure_region_offsets'):
        raise ValueError(f'{type(env).__name__} does not implement measure_region_offsets() -- '
                         f'--region_offsets_from_motion is not supported for --env_type {opt.env_type}')
    print("Measuring region entry/exit offsets from real IK/motion-planned trajectories...")
    measured = env.measure_region_offsets()
    print(f"  transit  alpha/beta = {measured['transit']}")
    print(f"  transfer alpha/beta = {measured['transfer']}")

    # A past run's repair_region_offsets() widening (below, after plan_refinement()) has nowhere
    # to land if nothing ever reads it back -- merge any cached prior widening in now (union, same
    # "only ever widen" rule repair_region_offsets() itself uses) so a past violation's correction
    # actually carries forward instead of being silently re-measured away on the next run.
    cached = load_cached_offsets(type(env).__name__)
    if cached:
        print(f"  cached from a prior run: {cached}")
        for action_type in ('transit', 'transfer'):
            if action_type in cached:
                ca, cb = cached[action_type]
                ma, mb = measured[action_type]
                measured[action_type] = (min(ca, ma), max(cb, mb))
        print(f"  merged -> transit={measured['transit']}  transfer={measured['transfer']}")

    opt.transit_region_entry_offset = measured['transit'][0]
    opt.transfer_region_exit_offset = measured['transfer'][1]
    planned_region_offsets = measured  # kept for the repair comparison after plan_refinement()
    # measure_region_offsets() draws random numbers (grasp shuffling, placement sampling) from the
    # same global random state everything else below uses -- without re-seeding, that shifts which
    # grasp/placement Steps 1-3 later draw purely as a side effect of THIS flag being on, making a
    # --region_offsets_from_motion run's chosen arm path differ from a plain run's even though path
    # computation itself (Steps 1-3) is completely unaffected by this feature. Re-seeding puts
    # everything after this point back on the exact same random draws either way, so the only real
    # difference between the two runs is the schedule this flag produces, not incidental path noise.
    random.seed(opt.seed)
    np.random.seed(opt.seed)

if opt.naive_playback:
    if opt.env_type != 'exp_duration_conflict_rai':
        raise ValueError('--naive_playback is only defined for exp_duration_conflict_rai')
    from mm_drrt.utils.naive_duration_executor import run_naive_duration_demo
    run_naive_duration_demo(env, C, transit_duration=opt.transit_duration, transfer_duration=opt.transfer_duration,
                            num_base_samples=opt.num_base_samples, num_arm_samples=opt.num_arm_samples,
                            use_debug=opt.use_debug)
    if opt.use_gui:
        input("Naive duration demo complete. Press Enter to close...")
    disconnect(C)
    raise SystemExit(0)

# Planner preference order when --use_pddl_planner is set: Tamer (solves the UPF problem
# directly) first, then classical Fast Downward (via MA-PDDL compilation) if Tamer fails,
# then the environment's manual plan as the last resort. Mirrors main.py.
planner_used = None
if opt.pddl_problem_file:
    from mm_drrt.planner.pddl_file_planner import generate_plan_from_pddl_files, PDDLFilePlannerError
    from mm_drrt.planner.tamer_pddl_planner import TamerPlannerError
    from mm_drrt.planner.pddl_domain import get_domain_file_path

    domain_file = opt.pddl_domain_file or get_domain_file_path()
    print(f"Solving hand-edited PDDL files: domain={domain_file} problem={opt.pddl_problem_file}")
    try:
        plan, action_orders, obj_orders, init_order_constraints = generate_plan_from_pddl_files(
            env, domain_file, opt.pddl_problem_file, timeout=opt.pddl_timeout)
    except (PDDLFilePlannerError, TamerPlannerError) as e:
        print(f"\n{type(e).__name__}: {e}\n\nNo fallback for --pddl_problem_file -- fix the "
             f"PDDL text and re-run to see how the solver reacts.")
        disconnect(C)
        raise SystemExit(1)
    planner_used = f'Tamer (from PDDL files: {opt.pddl_problem_file})'
elif opt.use_pddl_planner:
    from mm_drrt.planner.tamer_pddl_planner import TamerPDDLPlanner, TamerPlannerError, has_tamer_pddl_support
    from mm_drrt.planner.pddl_planner import PDDLPlanner, PDDLPlannerError, PDDLTimeoutError, has_pddl_support

    if has_tamer_pddl_support(env):
        print("Trying Tamer planner (PDDL 2.1) to generate task plan...")
        try:
            planner = TamerPDDLPlanner(timeout=opt.pddl_timeout, transit_duration=opt.transit_duration,
                                       transfer_duration=opt.transfer_duration,
                                       region_mutex_enabled=opt.region_mutex_enabled,
                                       transit_region_entry_offset=opt.transit_region_entry_offset,
                                       transit_region_exit_offset=opt.transit_region_exit_offset,
                                       transfer_region_entry_offset=opt.transfer_region_entry_offset,
                                       transfer_region_exit_offset=opt.transfer_region_exit_offset)
            plan, action_orders, obj_orders, init_order_constraints = planner.generate_plan(env)
            planner_used = 'Tamer (PDDL 2.1)'
        except TamerPlannerError as e:
            print(f"Tamer planning failed: {e}")
    else:
        print(f"Warning: Environment {opt.env_type} does not support Tamer PDDL planning")

    if planner_used is None:
        print("Falling back to classical Fast Downward planner...")
        if not has_pddl_support(env):
            print(f"Warning: Environment {opt.env_type} does not support PDDL planning")
        else:
            planner = PDDLPlanner(timeout=opt.pddl_timeout)
            try:
                plan, action_orders, obj_orders, init_order_constraints = planner.generate_plan(env)
                planner_used = 'Fast Downward (classical PDDL)'
            except PDDLTimeoutError as e:
                print(f"PDDL planner timeout: {e}")
            except PDDLPlannerError as e:
                print(f"PDDL planning failed: {e}")

    if planner_used is None:
        print("Falling back to manual plan specification...")
        plan, action_orders, obj_orders, init_order_constraints = env.create_plan_order_constraints()
        planner_used = 'manual (create_plan_order_constraints)'
else:
    plan, action_orders, obj_orders, init_order_constraints = env.create_plan_order_constraints()
    planner_used = 'manual (create_plan_order_constraints)'

print(f"Planner used: {planner_used}")

assert opt.num_robots == len(action_orders), "Error: num_robots is not properly set"
ps = PlanSkeleton(env, plan, obj_orders, init_order_constraints, opt.num_placement_samples, opt.use_debug)
refinement_start = time.time()
composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters, opt.drrt_time_limit)
refinement_elapsed = time.time() - refinement_start
print(f"Plan refinement (Steps 1-4) took {refinement_elapsed:.2f}s wall-clock; "
     f"composite path has {len(composite_path)} waypoints")
data_saver(composite_path, opt)

# Temporal repair (docs/minimal_temporal_constraints.md's "Future work"): compare what each
# region's mutex window was PLANNED with against what the REAL, dRRT*-refined composite path
# actually measured for that same run, via mm_drrt/utils/temporal_repair.py.
if opt.region_offsets_from_motion:
    if not hasattr(env, 'measure_executed_region_offsets'):
        print(f"Note: {type(env).__name__} does not implement measure_executed_region_offsets() "
             f"-- skipping temporal repair check for this run.")
    else:
        print("Measuring ACTUAL region entry/exit offsets from the executed composite path...")
        executed = env.measure_executed_region_offsets(composite_path)
        repaired = {}
        any_violation = False
        for action_type in ('transit', 'transfer'):
            planned = planned_region_offsets.get(action_type)
            measured_exec = executed.get(action_type)
            if planned is None or measured_exec is None:
                print(f"  {action_type}: nothing to repair against this run "
                     f"(no {'planned' if planned is None else 'executed'} measurement)")
                continue
            pa, pb = planned
            ma, mb = measured_exec
            result = repair_region_offsets(pa, pb, ma, mb)
            repaired[action_type] = (result.updated_alpha, result.updated_beta)
            status = "VIOLATED" if result.violated else "ok"
            print(f"  {action_type}: planned=({pa:.3f},{pb:.3f}) measured=({ma:.3f},{mb:.3f}) "
                 f"-> {status}  (entry_diff={result.entry_diff:+.3f}, exit_diff={result.exit_diff:+.3f})")
            if result.violated:
                any_violation = True
                # Decision: print loudly, don't fail/abort the run. dRRT*'s own
                # inter_robots_collision_fn (rai_drrt_star.py) already independently verified THIS
                # run's composite path is collision-free, regardless of what the PDDL-level mutex
                # window assumed -- same "about matching the solver's chosen concurrency, not about
                # safety" property mm_drrt/utils/pddl_parser.py's extract_shared_surface_constraints
                # already documents for its own ordering constraints. A violation here means the
                # offsets THIS run's schedule was planned around under-covered the real geometry --
                # a data-quality problem for the NEXT plan built from them, not evidence this run
                # was unsafe. Failing/asserting would abort an already-verified-safe run over a
                # background bookkeeping estimate; widening + persisting (below) is what actually
                # addresses it going forward.
                print(f"  ⚠ TEMPORAL REPAIR VIOLATION in {action_type}: the planned mutex "
                     f"window did not cover this run's real measured occupancy by more than "
                     f"the {DEFAULT_TOLERANCE}s tolerance. This run's safety was still "
                     f"independently guaranteed by dRRT*'s own collision checking; this is a "
                     f"signal to correct future planning, not that this run collided.")
        if repaired:
            cache_path = save_cached_offsets(type(env).__name__, repaired)
            print(f"  Saved widened offsets to {cache_path} for future runs.")
        if any_violation:
            print("WARNING: at least one region's temporal repair check was VIOLATED this run "
                 "(see above) -- the stored offsets have been widened to correct for it.")

if opt.use_gui:
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
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
    replay_start = time.time()
    replay_composite_path(C, composite_path, joints, release_targets, gripper_frames=gripper_frames)
    replay_elapsed = time.time() - replay_start
    # This is the number that should actually shift between --region_mutex_enabled runs and a
    # stock run: more overlap in the solved plan means fewer total composite-path waypoints to
    # step through, so a shorter playback here -- unlike the arms' own paths (Steps 1-3), which
    # this feature never touches (see main_rai.py's --region_offsets_from_motion reseed comment).
    print(f"Simulation playback took {replay_elapsed:.2f}s wall-clock "
         f"({len(composite_path)} composite waypoints)")
    input("Simulation complete. Press Enter to close...")
disconnect(C)
