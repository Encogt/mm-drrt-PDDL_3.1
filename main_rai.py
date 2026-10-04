import argparse
import random
import time
import numpy as np

from mm_drrt.utils.rai_utils import connect, disconnect, refresh_view
from mm_drrt.pipeline_rai import build_parser, make_env, measure_planned, run_pipeline, \
    release_targets_and_grippers
from mm_drrt.utils.rai_motion_planner_utils import replay_composite_path
from experiments.data_saver import data_saver

opt = build_parser().parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)

C = connect(use_gui=opt.use_gui)
env = make_env(opt, C)
refresh_view(C, use_gui=opt.use_gui)

planned_region_offsets, cache_key = measure_planned(env, opt)

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

result = run_pipeline(env, opt, planned_region_offsets, cache_key)
plan, composite_path = result.plan, result.composite_path

data_saver(composite_path, opt)

if opt.use_gui:
    robots = list(env.robots.values())
    joints = env.get_joints(robots)
    release_targets, gripper_frames = release_targets_and_grippers(env, plan)
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

