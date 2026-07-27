import argparse
import random
import numpy as np

from mm_drrt.utils.rai_utils import connect, disconnect, set_camera_pose, refresh_view

from examples.envs.example_single_robot_rai_env import ExampleSingleRobotRaiEnvironment, \
    ExampleSingleRobotCameraSetup
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

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

C = connect(use_gui=opt.use_gui)
if opt.env_type == 'exp_single_robot_rai':
    set_camera_pose(C, camera_point=ExampleSingleRobotCameraSetup[0], target_point=ExampleSingleRobotCameraSetup[1])
    env = ExampleSingleRobotRaiEnvironment(num_robots=opt.num_robots, num_objs=opt.num_objs, arm=opt.arm,
                                           grasp_type=opt.grasp_type, sim_id=C, seed=opt.seed)
else:
    raise ValueError('Unsupported env_type for the RAI POC: {}'.format(opt.env_type))

refresh_view(C, use_gui=opt.use_gui)

plan, action_orders, obj_orders, init_order_constraints = env.create_plan_order_constraints()

assert opt.num_robots == len(action_orders), "Error: num_robots is not properly set"
ps = PlanSkeleton(env, plan, obj_orders, init_order_constraints, opt.num_placement_samples, opt.use_debug)
composite_path = ps.plan_refinement(opt.num_base_samples, opt.num_arm_samples, opt.drrt_num_iters, opt.drrt_time_limit)
data_saver(composite_path, opt)

if opt.use_gui:
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
    # Single-object POC: whatever gets released was always headed for the last fixed obj (table1).
    release_targets = [env.f_objs[-1] for _ in robots]
    replay_composite_path(C, composite_path, joints, release_targets)
    input("Simulation complete. Press Enter to close...")
disconnect(C)
