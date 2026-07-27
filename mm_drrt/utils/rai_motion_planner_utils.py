"""RAI-flavored port of mm_drrt.utils.motion_planner_utils's orchestration layer (base_motion,
arm_retrieval_motion, the IK/IR/grasp generators, sub_plan_joint_motion, get_inter_robots_collision_fn).

Control flow mirrors the pybullet version function-for-function; only the leaf calls are swapped for
mm_drrt.utils.rai_utils. The pure graph/dRRT* algorithm helpers (TreeNode, OptimalNode, roadmap
bookkeeping, etc.) have no pybullet dependency at all and are reused directly from
motion_planner_utils instead of being duplicated here.
"""
from itertools import islice
import random
import time

import numpy as np

from mm_drrt.utils import rai_utils as ru
from mm_drrt.planner.prm import DegreePRM
from mm_drrt.planner.prm import prm
from mm_drrt.utils.motion_planner_utils import get_max_length_list

GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH


def get_trivial_roadmap(conf, attachments=[]):
    """A degenerate, no-op roadmap: start == goal == conf, nothing to actually plan. The
    multi-robot dRRT* machinery (task_planner_utils.py's assign_order_constraints, subprob_id
    indexing, etc.) hardcodes "3 roadmap segments per action" -- base motion, arm approach, arm
    retrieval, matching the mobile-base scenario's compute_path -- so a fixed-arm env (no base
    phase at all) must still contribute a 3rd placeholder roadmap per action rather than just 2,
    or that indexing arithmetic silently misaligns between robots.

    conf must be a real len(joints)-dimensional tuple, not (): get_substarts_subgoals() and
    friends slice the flat per-robot conf history with a *fixed stride* of
    dRRTStar.joint_dim = len(env.get_joints(robots)[0]) per subprob_id step -- a 0-length
    placeholder would desync that indexing for every step after it. The robot's own carry_conf is
    a natural choice (a real, definitely-safe conf); "base motion" is from carry to carry.

    attachments must match whatever the robot is actually holding while idling through this
    placeholder (e.g. [m_obj] for the gap between a transit that just grasped and the transfer
    that carries it onward) -- NOT always empty. get_subattachments() (motion_planner_utils.py,
    shared/unchanged) looks up "the segment just completed" as roadmaps[r][subprob_id[r] - 1]
    whenever subprob_id[r] changed since the last tree node, assuming that always means a single
    real step. _advance_past_trivial() (rai_drrt_star.py) can legitimately jump subprob_id by 2 in
    one go to skip over a trivial segment, which makes that lookback land ON the trivial segment
    instead of the real one before it -- so if this roadmap doesn't carry the correct holding
    state itself, get_subattachments() under-reports it (attach happens late, visibly desynced
    from the arm during replay) for that one transition.

    Since start==goal, PRM.__call__ (external/pybullet_planning/motion/motion_planners/prm.py)
    returns immediately without ever needing real sample/distance/extend/collision behavior --
    these are unreachable no-ops."""
    roadmap = DegreePRM(conf, conf, lambda: conf, lambda q1, q2: 0.0, lambda q1, q2: iter([conf]),
                       lambda q, verbose=False: False, samples=[conf], target_degree=0,
                       expand_type=None, expand_configs=(), attachments=attachments)
    roadmap.final_conf = conf
    heuristic_val = {roadmap.vertices[conf]: 0.0}
    return roadmap, heuristic_val


def replay_composite_path(C, composite_path, joints, release_targets, gripper_frames=None, pause_time=0.02):
    """Step C through a solved composite_path (list of OptimalNode, from PlanSkeleton.plan_refinement)
    in the live viewer. Neither this pipeline nor the pybullet original animates the plan on its own
    -- the pybullet version needs a separate experiments/visualizer.py run against the pickled
    output -- so this replays it directly in the same window used for planning.

    joints: per-robot list of joint names (env.get_joints(robots)).
    release_targets: per-robot frame name to reattach a carried object to once it's released (the
    composite_path only records the grasped object's name, not its destination surface, so the
    caller supplies it -- for this single-object POC it's simply the destination table).
    gripper_frames: per-robot gripper frame name to attach a carried object to. Defaults to
    ru.GRIPPER_FRAME for every robot (the single-mobile-robot scenario).
    """
    num_robots = len(joints)
    if gripper_frames is None:
        gripper_frames = [ru.GRIPPER_FRAME] * num_robots
    currently_attached = [None] * num_robots
    for node in composite_path:
        for j in range(get_max_length_list(node.sub_local_paths)):
            for r in range(num_robots):
                path_r = node.sub_local_paths[r]
                if not path_r:
                    continue
                q = path_r[-1] if len(path_r) <= j else path_r[j]
                ru.set_joint_positions(C, joints[r], q)
            for r in range(num_robots):
                held = node.attachments[r] if node.attachments else None
                if held and not currently_attached[r]:
                    C.attach(gripper_frames[r], held)
                    currently_attached[r] = held
                elif not held and currently_attached[r]:
                    C.attach(release_targets[r], currently_attached[r])
                    currently_attached[r] = None
            C.view(False)
            time.sleep(pause_time)


def plan_joint_motion(robot, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=None,
                      use_aabb=False, cache=True, custom_limits={}, use_drrt_star=False,
                      use_debug_plot=False, **kwargs):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = ru.get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = ru.get_distance_fn(robot, joints, weights=weights)
    extend_fn = ru.get_extend_fn(robot, joints, resolutions=resolutions)
    collision_fn = ru.get_collision_fn(robot, joints, obstacles, attachments, self_collisions,
                                       disabled_collisions, custom_limits=custom_limits)

    start_conf = ru.get_joint_positions(robot, joints)
    if not ru.check_initial_end(start_conf, end_conf, collision_fn):
        return None
    return prm(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
              use_drrt_star=use_drrt_star, use_debug_plot=use_debug_plot, **kwargs)


def base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[], custom_limits={},
                num_samples=100, expand_type=None, expand_configs=None, use_debug=False):
    base_joints = ru.BASE_JOINTS
    ru.set_joint_positions(robot, base_joints, base_start)
    base_goal = base_goal[:len(base_joints)]

    if teleport:
        ru.set_joint_positions(robot, base_joints, base_start)
        if any(ru.robot_obstacle_collision(robot, [b]) for b in obstacles):
            return None
        ru.set_joint_positions(robot, base_joints, base_goal)
        if any(ru.robot_obstacle_collision(robot, [b]) for b in obstacles):
            return None
        return [base_start, base_goal]

    resolutions = np.array([2 * ru.DEFAULT_BASE_RESOLUTION, 2 * ru.DEFAULT_BASE_RESOLUTION,
                            ru.DEFAULT_BASE_RESOLUTION])
    for _ in range(5):
        roadmap, heuristic_val = sub_plan_joint_motion(robot, base_joints, base_goal, obstacles=obstacles,
                                                       attachments=attachments, resolutions=resolutions,
                                                       custom_limits=custom_limits, use_drrt_star=True,
                                                       num_samples=num_samples, expand_type=expand_type,
                                                       expand_configs=expand_configs, use_debug=use_debug)
        if roadmap is None:
            num_samples += 20
            ru.set_joint_positions(robot, base_joints, base_start)
        else:
            break
    if not roadmap:
        ru.set_joint_positions(robot, base_joints, base_start)
    ru.set_joint_positions(robot, base_joints, base_goal)
    return roadmap, heuristic_val


def arm_retrieval_motion(robot, arm, type, grasp, start, goal, obstacles=[], attachments=[], custom_limits={},
                         num_samples=100, expand_type=None, expand_configs=None, use_debug=False,
                         arm_joints=None):
    arm_joints, _, _, _ = ru._spec_of(robot, arm_joints)
    ru.set_joint_positions(robot, arm_joints, start)
    resolutions = 0.05 ** np.ones(len(arm_joints))
    for _ in range(5):
        roadmap, heuristic_val = sub_plan_joint_motion(robot, arm_joints, goal, obstacles=obstacles,
                                                       attachments=attachments, self_collisions=False,
                                                       resolutions=resolutions, custom_limits=custom_limits,
                                                       use_drrt_star=True, num_samples=num_samples,
                                                       expand_type=expand_type, expand_configs=expand_configs,
                                                       use_debug=use_debug)
        if roadmap is None:
            num_samples += 20
            ru.set_joint_positions(robot, arm_joints, start)
        else:
            break
    if not roadmap:
        ru.set_joint_positions(robot, arm_joints, start)
        return None, None
    ru.set_joint_positions(robot, arm_joints, goal)
    return roadmap, heuristic_val


def get_ir_sampler(robot, gripper, custom_limits={}, max_attempts=25, collisions=True, collision_objs=[],
                   num_samples=100, learned=True, use_debug=False):
    # Note: 'learned' (PR2 IK-reachability model) has no RAI equivalent; always uniform-sample the
    # base pose near the target, as documented in the plan.
    obstacles = collision_objs if collisions else []

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        target_xy = ru.get_frame_position(robot, obj)[:2]
        base_joints = ru.BASE_JOINTS
        lower_limits, upper_limits = ru.get_custom_limits(robot, base_joints, custom_limits)
        base_generator = ru.uniform_pose_generator(robot, target_xy)
        while True:
            for base_conf in islice(base_generator, max_attempts):
                if not ru.all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = ru.Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                ru.set_joint_positions(robot, ru.ARM_JOINTS, grasp.carry)
                if any(ru.robot_obstacle_collision(robot, [b]) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None

    return gen_fn


def get_grasp_gen(grasp_type, collisions=False, randomize=True):
    def fn(robot, body):
        if grasp_type == 'top':
            grasps = ru.get_top_grasps(body)
        elif grasp_type == 'side':
            grasps = ru.get_side_grasps(body)
        else:
            raise ValueError('Unexpected grasp type:', grasp_type)
        if randomize:
            random.shuffle(grasps)
        return [(g,) for g in grasps]

    return fn


def get_placement_gen(robot, **kwargs):
    def gen(body, surface):
        while True:
            body_pose = ru.sample_placement(robot, body, surface, **kwargs)
            if body_pose is None:
                break
            p = ru.Pose(robot, body, body_pose, surface)
            yield (p,)
    return gen


def get_stable_gen(robot, collisions=True, **kwargs):
    def gen(body, surface, obstacles):
        while True:
            body_pose = ru.sample_placement(robot, body, surface, **kwargs)
            if body_pose is None:
                break
            p = ru.Pose(robot, body, body_pose, surface)
            p.assign()
            if not any(ru.pairwise_collision(robot, body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    return gen


def get_ik_fn(robot, custom_limits={}, collisions=True, collision_objs=[], use_debug=False, arm_joints=None):
    obstacles = collision_objs if collisions else []
    arm_joints, _, _, _ = ru._spec_of(robot, arm_joints)

    def fn(arm, obj, pose, grasp, base_conf):
        pose.assign()
        if base_conf is not None:
            base_conf.assign()
        ru.set_joint_positions(robot, arm_joints, grasp.carry)
        base_values = base_conf.values if base_conf is not None else None
        grasp_conf = ru.rai_ik(robot, base_values, grasp, custom_limits=custom_limits, arm_joints=arm_joints)
        if grasp_conf is None:
            if use_debug:
                print('Grasp IK failure: no IK solution')
            return (None, None)
        ru.set_joint_positions(robot, arm_joints, grasp_conf)
        # Validate both self-collision (rai_ik solves with enableCollisions=False, so a solution can
        # be self-colliding, e.g. wrist twisted into the forearm) and collision against obstacles.
        # obj itself is excluded: at the grasp conf the gripper is expected to touch/enclose it.
        if ru.check_collisions(robot, obstacles, self_collisions=True, verbose=use_debug):
            if use_debug:
                print('Grasp IK failure: colliding solution', grasp_conf)
            return (None, None)
        # Approach conf: unlike pybullet-planning's separate sub_inverse_kinematics call to an offset
        # approach pose, we reuse the grasp IK result as the approach target too (retrieval motion
        # supplies the actual collision-checked path between them) -- sufficient to prove the
        # pipeline end-to-end.
        approach_conf = grasp_conf
        return (approach_conf, grasp_conf)

    return fn


def get_arm_motion_fn(robot, custom_limits={}, collisions=True, collision_objs=[], num_samples=100,
                      expand_type=None, expand_configs=None, use_debug=False, arm_joints=None):
    obstacles = collision_objs if collisions else []
    arm_joints, _, _, _ = ru._spec_of(robot, arm_joints)

    def fn(arm, obj, grasp, approach_conf, grasp_conf, attachments=[]):
        roadmap, heuristic_val = None, None
        resolutions = 0.05 ** np.ones(len(arm_joints))
        ru.set_joint_positions(robot, arm_joints, grasp.carry)
        _num_samples = num_samples
        for _ in range(5):
            roadmap, heuristic_val = sub_plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments,
                                                           obstacles=obstacles, self_collisions=True,
                                                           custom_limits=custom_limits, resolutions=resolutions,
                                                           use_drrt_star=True, num_samples=_num_samples,
                                                           expand_type=expand_type, expand_configs=expand_configs,
                                                           use_debug=use_debug)
            if roadmap is None:
                _num_samples += 20
                ru.set_joint_positions(robot, arm_joints, grasp.carry)
            else:
                break
        if roadmap is None:
            if use_debug:
                print('Roadmap is not found')
            ru.set_joint_positions(robot, arm_joints, grasp.carry)
            return (None, None)
        approach_path = roadmap(expand_configs + tuple(grasp.carry), expand_configs + tuple(approach_conf))
        if approach_path is None:
            if use_debug:
                print('Approach path failure')
            return (None, None)
        ru.set_joint_positions(robot, arm_joints, approach_conf)
        grasp_path = ru.plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments,
                                                 obstacles=obstacles, self_collisions=True,
                                                 custom_limits=custom_limits, resolutions=resolutions / 2.)
        if grasp_path is None:
            if use_debug:
                print('Grasp path failure')
            return (None, None)
        ru.set_joint_positions(robot, arm_joints, grasp_conf)

        grasp_conf = expand_configs + tuple(grasp_conf)
        grasp_path = [expand_configs + tuple(g) for g in grasp_path]
        roadmap.add([grasp_conf])
        roadmap.connect(roadmap.vertices[roadmap.final_conf], roadmap.vertices[grasp_conf], grasp_path)
        roadmap.final_conf = grasp_conf
        for v in heuristic_val.keys():
            heuristic_val[v] += 1.0
        heuristic_val[roadmap.vertices[grasp_conf]] = 0.0
        return (roadmap, heuristic_val)

    return fn


def get_ik_ir_gen(robot, gripper, max_attempts=25, learned=True, use_debug=False, **kwargs):
    ir_sampler = get_ir_sampler(robot, gripper, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(robot, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if use_debug:
                print('attempts', attempts)
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None or ik_outputs[0] is None:
                continue
            if use_debug:
                print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return

    return gen


def get_fixed_arm_ik_ir_gen(robot, max_attempts=25, use_debug=False, **kwargs):
    """Fixed-arm analogue of get_ik_ir_gen, for a robot with no base joints (the arm is bolted
    down, so there's nothing to search over the way get_ir_sampler searches candidate base poses
    -- the only real search is the grasp IK itself, which rai_ik already random-restarts
    internally). Still yields the same (bq, approach_conf, grasp_conf) 3-tuple shape get_ik_ir_gen
    does -- bq is a trivial Conf over zero joints -- so callers (subgoal_sampling, compute_path)
    don't need a different code path for the fixed-arm vs mobile-base scenario."""
    ik_fn = get_ik_fn(robot, use_debug=use_debug, **kwargs)

    def gen(arm, obj, pose, grasp):
        bq = ru.Conf(robot, [], ())
        for attempts in range(max_attempts):
            if use_debug:
                print('attempts', attempts)
            ik_outputs = ik_fn(arm, obj, pose, grasp, None)
            if ik_outputs is None or ik_outputs[0] is None:
                continue
            if use_debug:
                print('IK attempts:', attempts)
            yield (bq,) + ik_outputs
            return
        yield None

    return gen


def get_gripper(robot_id, arm='left', visual=True):
    _, gripper_frame, _, _ = ru._spec_of(robot_id)
    return gripper_frame


def get_configuration(robot, name):
    arm_joints, _, base_joints, _ = ru._spec_of(robot)
    if name == 'base':
        return ru.get_joint_positions(robot, base_joints)
    return ru.get_joint_positions(robot, arm_joints)


def get_arm_positions(robot, arm):
    arm_joints, _, _, _ = ru._spec_of(robot)
    return ru.get_joint_positions(robot, arm_joints)


def sub_plan_joint_motion(robot, joints, end_conf, obstacles=[], attachments=[],
                          self_collisions=True, disabled_collisions=set(),
                          weights=None, resolutions=None, max_distance=None,
                          use_aabb=False, cache=True, custom_limits={}, use_drrt_star=False,
                          use_debug_plot=False, **kwargs):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sub_sample_fn = ru.get_sample_fn(robot, joints, custom_limits=custom_limits)
    sub_distance_fn = ru.get_distance_fn(robot, joints, weights=weights)
    sub_extend_fn = ru.get_extend_fn(robot, joints, resolutions=resolutions)
    sub_collision_fn = ru.get_collision_fn(robot, joints, obstacles, attachments, self_collisions,
                                           disabled_collisions, custom_limits=custom_limits)

    start_conf = ru.get_joint_positions(robot, joints)
    if not ru.check_initial_end(start_conf, end_conf, sub_collision_fn):
        return None, None
    from mm_drrt.utils.motion_planner_utils import debug_2d_sub_roadmap
    return prm(start_conf, end_conf, sub_distance_fn, sub_sample_fn, sub_extend_fn, sub_collision_fn,
              attachments=attachments, use_drrt_star=use_drrt_star, use_debug_plot=use_debug_plot,
              debug_roadmap_fn=debug_2d_sub_roadmap, **kwargs)


def get_inter_robots_collision_fn(robots, joints, num_robots=1, tolerance=None, **kwargs):
    """dRRT*'s composite-state collision check between distinct robots. Multiple robots can share
    one ry.Config (the fixed dual-arm scenario) or each have their own; either way, every robot
    that's a RaiRobot carries its own spec.frame_prefix, which is how a colliding frame pair gets
    attributed to two DIFFERENT robots (a pair sharing one robot's prefix is that robot's own
    self-collision, already checked elsewhere -- not this function's job)."""
    if tolerance is None:
        tolerance = ru.COLLISION_TOLERANCE
    prefixes = [(r.spec.frame_prefix if isinstance(r, ru.RaiRobot) else None) for r in robots]

    def _cross_robot_pair(a, b):
        for i in range(num_robots):
            for j in range(num_robots):
                if i == j or not prefixes[i] or not prefixes[j]:
                    continue
                if a.startswith(prefixes[i]) and b.startswith(prefixes[j]):
                    return i, j
        return None

    def drrt_collision_fn(q, mode='boolean', skip_index=[]):
        for r in range(num_robots):
            ru.set_joint_positions(robots[r], joints[r], q[r])

        seen, collisions = set(), []
        for r in range(num_robots):
            C = ru._config_of(robots[r])
            if id(C) in seen:
                continue
            seen.add(id(C))
            C.computeCollisions()
            collisions.extend(C.getCollisions())

        if mode == 'boolean':
            for a, b, pen in collisions:
                if pen >= -tolerance:
                    continue
                if _cross_robot_pair(a, b):
                    return True
            return False
        elif mode == 'index':
            collision_robot_index = []
            for a, b, pen in collisions:
                if pen >= -tolerance:
                    continue
                pair = _cross_robot_pair(a, b)
                if pair and pair[0] not in skip_index and pair[1] not in skip_index:
                    collision_robot_index.extend(pair)
            return list(np.unique(collision_robot_index))
    return drrt_collision_fn
