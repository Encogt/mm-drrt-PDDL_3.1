import argparse
import random
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
from mm_drrt.planner.rai_task_planner import PlanSkeleton
from mm_drrt.utils.rai_motion_planner_utils import replay_composite_path
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
else:
    raise ValueError('Unsupported env_type for the RAI POC: {}'.format(opt.env_type))

refresh_view(C, use_gui=opt.use_gui)

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
                                       transfer_duration=opt.transfer_duration)
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
composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters, opt.drrt_time_limit)
data_saver(composite_path, opt)

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
    replay_composite_path(C, composite_path, joints, release_targets, gripper_frames=gripper_frames)
    input("Simulation complete. Press Enter to close...")
disconnect(C)
