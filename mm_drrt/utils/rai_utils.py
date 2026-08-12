"""RAI (robotic / ry.Config) replacement for the pybullet-planning primitives used by
mm_drrt.utils.motion_planner_utils. See /home/enco/.claude/plans/deep-wobbling-sonnet.md for the
design this implements.

Robot: RAI's built-in 'scenarios/panda_ranger.g' -- a 7-DOF Panda arm ('l_panda_joint1..7') on a
3-DOF planar Ranger base ('ranger_transX', 'ranger_transY', 'ranger_rot'), with a gripper reference
frame 'l_gripper'. This plays the role PR2 + its 'left'/'base' joint groups played in the pybullet
version.

Grasping is expressed with KOMO features (negDistance + scalarProduct alignment) against the target
object's own frame, rather than porting pybullet-planning's explicit SE3 grasp-transform algebra.
Grasped objects are carried via RAI's native C.attach(), so -- unlike the pybullet Attachment class --
nothing needs to be re-applied per sampled configuration; once attached, every set_joint_positions()
call keeps the object correctly posed relative to the gripper for free.
"""
import math
import random
import time

import numpy as np
import robotic as ry

INF = float('inf')

BASE_JOINTS = ['ranger_transX', 'ranger_transY', 'ranger_rot']
ARM_JOINTS = [f'l_panda_joint{i}' for i in range(1, 8)]
GRIPPER_FRAME = 'l_gripper'
ROBOT_SCENARIO = 'scenarios/panda_ranger.g'
ROBOT_FRAME_PREFIXES = ('ranger_', 'l_panda', 'l_finger', 'l_palm', 'l_gripper',
                        'r_panda', 'r_finger', 'r_palm', 'r_gripper')

# Reasonable folded/carry arm posture (within joint limits), used as the "resting" configuration
# before/after a pick or place, analogous to PR2's TOP_HOLDING_LEFT_ARM / SIDE_HOLDING_LEFT_ARM.
# This is Franka's standard self-collision-free "ready" pose (verified: no self-collision, and
# clears a table at the base standoff radii uniform_pose_generator samples). Also verified
# self-collision-free -- and cross-arm-collision-free -- for the fixed dual-arm scenario below.
PANDA_CARRY_CONF = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

DEFAULT_ARM_RESOLUTION = math.radians(3)
DEFAULT_BASE_RESOLUTION = 0.02
COLLISION_TOLERANCE = 5e-3


def is_robot_frame(name):
    return name.startswith(ROBOT_FRAME_PREFIXES)


##### Multi-robot identity (see plan: RaiRobot wrapper) #####

class ArmSpec(object):
    """Everything that's specific to one arm sharing a Config with other robots: its own joint
    names, gripper reference frame, collision-frame prefix (for inter-robot collision
    attribution), and carry/rest posture. base_joints is empty for a fixed-mount arm."""

    def __init__(self, arm_joints, gripper_frame, frame_prefix, carry_conf, base_joints=()):
        self.arm_joints = list(arm_joints)
        self.gripper_frame = gripper_frame
        self.frame_prefix = frame_prefix
        self.carry_conf = tuple(carry_conf)
        self.base_joints = list(base_joints)


class RaiRobot(object):
    """Identifies one robot/arm sharing a ry.Config with others. mm_drrt's multi-robot machinery
    (action_orders, robot_plans in task_planner_utils.py) uses 'robot' as a dict key, which
    requires each robot to be a distinct hashable object -- but two arms on one Config can't
    both just *be* that Config (self.robots[0] is self.robots[1] would collapse to one key).
    RaiRobot gives each arm its own identity (default identity-based __eq__/__hash__) while
    still carrying a reference to the shared Config and this arm's ArmSpec."""

    def __init__(self, C, spec: ArmSpec):
        self.C = C
        self.spec = spec

    def __repr__(self):
        return 'RaiRobot({})'.format(self.spec.frame_prefix or 'default')


def _config_of(robot):
    """robot is either a bare ry.Config (the single-mobile-robot scenario, where the Config IS
    the robot identity) or a RaiRobot wrapping one (the multi-arm scenario). Unwrap to the
    actual Config either way."""
    return robot.C if isinstance(robot, RaiRobot) else robot


def _spec_of(robot, arm_joints=None, gripper_frame=None, base_joints=None, carry_conf=None):
    """Resolve the (arm_joints, gripper_frame, base_joints, carry_conf) to use for this robot:
    explicit arguments win; otherwise use robot.spec if it's a RaiRobot; otherwise fall back to
    the single-mobile-robot module constants (today's default behavior, unchanged)."""
    spec = robot.spec if isinstance(robot, RaiRobot) else None
    return (
        arm_joints if arm_joints is not None else (spec.arm_joints if spec else ARM_JOINTS),
        gripper_frame if gripper_frame is not None else (spec.gripper_frame if spec else GRIPPER_FRAME),
        base_joints if base_joints is not None else (spec.base_joints if spec else BASE_JOINTS),
        carry_conf if carry_conf is not None else (spec.carry_conf if spec else PANDA_CARRY_CONF),
    )


TWO_ARM_SCENARIO = 'scenarios/pandasTable.g'
LEFT_ARM = ArmSpec([f'l_panda_joint{i}' for i in range(1, 8)], 'l_gripper', 'l_', PANDA_CARRY_CONF)
RIGHT_ARM = ArmSpec([f'r_panda_joint{i}' for i in range(1, 8)], 'r_gripper', 'r_', PANDA_CARRY_CONF)


##### Session #####

def connect(use_gui=True):
    C = ry.Config()
    return C


def refresh_view(C, use_gui=True):
    """Call once after the scene (robot/tables/objects) has been fully built. The viewer needs
    view_recopyMeshes() to pick up meshes added via addFile()/addFrame() after it was opened --
    without it the window comes up but stays empty ("no robot"), since view() only shows whatever
    was already in C at the time it was first called."""
    if not use_gui:
        return
    C.view_recopyMeshes()
    C.view(False, 'mm-drrt (RAI)')


def disconnect(C):
    try:
        C.view_close()
    except AttributeError:
        pass


def set_camera_pose(C, camera_point=None, target_point=None):
    # RAI's default viewer camera is adequate for the POC; nothing else to wire up.
    return


##### Joint state #####

_joint_index_cache = {}


def _joint_indices(C, joint_names):
    C = _config_of(C)
    key = id(C)
    all_names = _joint_index_cache.get(key)
    if all_names is None or all_names != list(C.getJointNames()):
        all_names = list(C.getJointNames())
        _joint_index_cache[key] = all_names
    return [all_names.index(n) for n in joint_names]


def get_joint_positions(C, joint_names):
    C = _config_of(C)
    idx = _joint_indices(C, joint_names)
    return tuple(np.asarray(C.getJointState())[idx])


def set_joint_positions(C, joint_names, values):
    C = _config_of(C)
    C.setJointState(np.asarray(values, dtype=float), list(joint_names))


def get_joint_limits(C, joint_names):
    C = _config_of(C)
    idx = _joint_indices(C, joint_names)
    limits = C.getJointLimits()
    lower = np.asarray(limits[0])[idx]
    upper = np.asarray(limits[1])[idx]
    return lower, upper


def get_custom_limits(C, joint_names, custom_limits={}):
    lower, upper = get_joint_limits(C, joint_names)
    lower, upper = list(lower), list(upper)
    for i, name in enumerate(joint_names):
        if name in custom_limits:
            lower[i], upper[i] = custom_limits[name]
    return lower, upper


##### Pose / Conf state objects (API mirrors pr2_primitives.Pose/Conf) #####

class Conf(object):
    def __init__(self, C, joint_names, values=None, init=False):
        self.C = _config_of(C)
        self.joints = list(joint_names)
        if values is None:
            values = get_joint_positions(C, joint_names)
        self.values = tuple(values)
        self.init = init

    def assign(self):
        set_joint_positions(self.C, self.joints, self.values)

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)


class Pose(object):
    _num = 0

    def __init__(self, C, frame_name, value=None, support=None, init=False):
        self.C = _config_of(C)
        self.frame_name = frame_name
        if value is None:
            f = self.C.getFrame(frame_name)
            value = (tuple(f.getPosition()), tuple(f.getQuaternion()))
        self.value = value
        self.support = support
        self.init = init
        Pose._num += 1
        self.index = Pose._num

    def assign(self):
        f = self.C.getFrame(self.frame_name)
        point, quat = self.value
        f.setPosition(point)
        f.setQuaternion(quat)

    def __repr__(self):
        return 'p{}'.format(self.index)


class Grasp(object):
    """A grasp 'style' expressed as KOMO objectives against the object's own frame, rather than a
    pre-computed SE3 gripper-from-object transform. grasp_type in {'top', 'side'} only changes which
    scalarProduct alignment objective is added at IK time (see rai_ik())."""

    def __init__(self, grasp_type, obj_frame_name, carry=PANDA_CARRY_CONF, approach=0.12):
        self.grasp_type = grasp_type
        self.obj_frame_name = obj_frame_name
        self.carry = tuple(carry)
        self.approach = approach
        self.grasp_width = 0.0

    def attach(self, C, gripper_frame=GRIPPER_FRAME):
        _config_of(C).attach(gripper_frame, self.obj_frame_name)

    def detach(self, C, new_parent_frame):
        _config_of(C).attach(new_parent_frame, self.obj_frame_name)

    def __repr__(self):
        return 'g({})'.format(self.grasp_type)


##### Sampling primitives (the 4 closures prm.py / motion_planner_utils.py depend on) #####

def get_sample_fn(C, joint_names, custom_limits={}):
    lower, upper = get_custom_limits(C, joint_names, custom_limits)
    lower, upper = np.asarray(lower), np.asarray(upper)

    def sample_fn():
        return tuple(np.random.uniform(lower, upper))

    return sample_fn


def get_distance_fn(C, joint_names, weights=None):
    if weights is None:
        weights = np.ones(len(joint_names))
    weights = np.asarray(weights)
    n = len(weights)

    def distance_fn(q1, q2):
        # prm.py's PRM.__call__() calls this on full expand_type-concatenated tuples (base+arm or
        # arm+base), not just this function's own len(joint_names) sub-range. The pybullet original
        # tolerated that via zip()'s implicit shortest-iterable truncation; replicate that here by
        # truncating both configs to the first n=len(weights) entries (matches the base-first /
        # base-prefix convention prm.py's expand_type branches use).
        diff = np.asarray(q2[:n]) - np.asarray(q1[:n])
        return math.sqrt(np.dot(weights, diff * diff))

    return distance_fn


def get_extend_fn(C, joint_names, resolutions=None):
    if resolutions is None:
        resolutions = np.array([
            DEFAULT_BASE_RESOLUTION if name in BASE_JOINTS else DEFAULT_ARM_RESOLUTION
            for name in joint_names
        ])
    resolutions = np.asarray(resolutions)

    def extend_fn(q1, q2):
        q1, q2 = np.asarray(q1), np.asarray(q2)
        diff = q2 - q1
        steps = int(np.max(np.abs(diff) / resolutions)) if len(diff) else 0
        steps = max(steps, 1)
        for i in range(1, steps + 1):
            yield tuple(q1 + diff * (float(i) / steps))

    return extend_fn


def all_between(lower, values, upper):
    return bool(np.less_equal(lower, values).all() and np.less_equal(values, upper).all())


def check_collisions(C, obstacles=[], attachments=[], self_collisions=True, tolerance=COLLISION_TOLERANCE,
                     verbose=False):
    """True if the CURRENT config state has any disallowed collision: robot self-collision (if
    self_collisions), or robot/attached-object penetrating a named obstacle frame. Does not touch
    joint state -- callers set it first. Shared by get_collision_fn (per-sample, PRM-facing) and
    rai_ik / get_ik_fn (so IK solutions are validated for self-collision too, not just obstacles).

    Note for the multi-arm scenario: is_robot_frame() doesn't distinguish which arm a frame
    belongs to, so if arm A's sampled pose overlaps arm B's *current* pose, this reports it as
    a "self-collision" (mislabeled, but functionally correct -- it still blocks the invalid
    configuration). Real arm-vs-arm attribution (for dRRT*'s composite search, which explores
    combinations never tested during each arm's own roadmap construction) is
    get_inter_robots_collision_fn in rai_motion_planner_utils.py."""
    C = _config_of(C)
    obstacle_names = set(obstacles)
    attached_names = set(attachments)
    C.computeCollisions()
    for a, b, pen in C.getCollisions():
        if pen >= -tolerance:
            continue
        a_robot, b_robot = is_robot_frame(a), is_robot_frame(b)
        a_att, b_att = a in attached_names, b in attached_names
        a_obs, b_obs = a in obstacle_names, b in obstacle_names
        if (a_att and b_robot) or (b_att and a_robot):
            continue  # object touching the gripper that is holding it: expected
        if a_att and b_att:
            continue
        if a_robot and b_robot:
            if self_collisions:
                if verbose:
                    print('Self-collision:', a, b, pen)
                return True
            continue
        if (a_robot or a_att) and b_obs:
            if verbose:
                print('Collision:', a, b, pen)
            return True
        if (b_robot or b_att) and a_obs:
            if verbose:
                print('Collision:', a, b, pen)
            return True
    return False


def get_collision_fn(C, joint_names, obstacles=[], attachments=[], self_collisions=True,
                     disabled_collisions=set(), custom_limits={}, max_distance=None,
                     use_aabb=False, cache=True, tolerance=COLLISION_TOLERANCE, **kwargs):
    lower, upper = get_custom_limits(C, joint_names, custom_limits)

    def collision_fn(q, verbose=False):
        if not all_between(lower, q, upper):
            if verbose:
                print('Joint limits violated:', q)
            return True
        set_joint_positions(C, joint_names, q)
        return check_collisions(C, obstacles, attachments, self_collisions, tolerance, verbose)

    return collision_fn


def check_initial_end(start_conf, end_conf, collision_fn, verbose=True):
    if collision_fn(start_conf, verbose=verbose):
        print('Warning: initial configuration is in collision')
        return False
    if collision_fn(end_conf, verbose=verbose):
        print('Warning: end configuration is in collision')
        return False
    return True


def pairwise_collision(C, name_a, name_b, tolerance=COLLISION_TOLERANCE):
    C = _config_of(C)
    C.computeCollisions()
    for a, b, pen in C.getCollisions():
        if pen >= -tolerance:
            continue
        if {a, b} == {name_a, name_b}:
            return True
    return False


def boxes_overlap(C, name_a, name_b, margin=0.0):
    """Axis-aligned bounding-box overlap check between two box-shaped frames, using their
    getPosition()/getSize() directly rather than C.computeCollisions(). Needed specifically for
    placement-sampling collision checks: RAI's collision engine appears to cache broadphase
    proximity from a config's initial frame layout and never re-detect a pair that started far
    apart (e.g. two movable objects' own starting surfaces, meters apart) even after both are
    later moved close together by sample_placement() -- confirmed by direct reproduction: two
    boxes moved to 3cm apart (well within their ~7x5cm footprint, a clear overlap) reported zero
    entries from computeCollisions()/getCollisions(), while the identical check against frames
    that started out close together works correctly. Every movable-object placement this project
    samples is axis-aligned (sample_placement() always uses an identity quaternion), so a plain
    AABB test is exact here, not an approximation."""
    C = _config_of(C)
    fa, fb = C.getFrame(name_a), C.getFrame(name_b)
    pa, pb = np.asarray(fa.getPosition()), np.asarray(fb.getPosition())
    sa, sb = np.asarray(fa.getSize()[:3]), np.asarray(fb.getSize()[:3])
    return bool(np.all(np.abs(pa - pb) < (sa + sb) / 2.0 + margin))


def robot_obstacle_collision(C, obstacle_names, tolerance=COLLISION_TOLERANCE):
    """True if any robot frame currently penetrates any of the named obstacle frames."""
    C = _config_of(C)
    obstacle_names = set(obstacle_names)
    C.computeCollisions()
    for a, b, pen in C.getCollisions():
        if pen >= -tolerance:
            continue
        a_robot, b_robot = is_robot_frame(a), is_robot_frame(b)
        if (a_robot and b in obstacle_names) or (b_robot and a in obstacle_names):
            return True
    return False


def plan_direct_joint_motion(C, joint_names, end_conf, obstacles=[], attachments=[],
                             self_collisions=True, custom_limits={}, resolutions=None, **kwargs):
    extend_fn = get_extend_fn(C, joint_names, resolutions=resolutions)
    collision_fn = get_collision_fn(C, joint_names, obstacles, attachments, self_collisions,
                                    custom_limits=custom_limits)
    start_conf = get_joint_positions(C, joint_names)
    path = [start_conf] + list(extend_fn(start_conf, end_conf))
    if any(collision_fn(q) for q in path):
        return None
    return path


##### Grasp-pose IK via KOMO #####

# Pairs of scalarProduct constraints, [gripper_frame, obj_frame_name] order, that together pin
# the gripper's local X axis (its squeeze axis) to be PARALLEL to one specific local axis of the
# object -- rather than just perpendicular to one axis (leaving a free rotation around the
# approach direction). Matches robotic/manipulation.py's own grasp_box(): its 'x'/'y'/'z'
# grasp_direction choices use exactly these three pairs to align the squeeze axis with the
# object's X/Y/Z axis respectively. 'top' aligns with the object's own X axis (a horizontal
# squeeze, matching an approach-from-above); 'side' aligns with the object's Z axis (a vertical
# squeeze, matching an approach-from-the-side).
_GRASP_ALIGN = {
    'top': (ry.FS.scalarProductXY, ry.FS.scalarProductXZ),
    'side': (ry.FS.scalarProductXX, ry.FS.scalarProductXY),
}

# Row selector (for a positionRel equality objective) picking out the object axes to center the
# gripper on -- negDistance alone only requires the gripper to touch the object SOMEWHERE, so
# without this a solve can converge anywhere from dead-center to right at the object's edge
# (confirmed empirically, one axis at a time: world-frame offset ranged +-0.035 along the squeeze
# axis and +-0.025 along the other in-plane axis, i.e. the object's full half-width each way,
# across repeated solves of the same grasp). Centers TWO of the object's three axes: the squeeze
# axis itself (matches _GRASP_ALIGN above) and the other in-plane axis -- leaving only the
# approach/depth axis unconstrained, since THAT one is where negDistance's touching-the-surface
# distance has to live (fully centering all three would force the gripper's marker to the
# object's own center, i.e. deep inside it, directly contradicting negDistance==0).
_GRASP_CENTER_AXES = {
    'top': np.array([[1, 0, 0], [0, 1, 0]]),
    'side': np.array([[0, 0, 1], [1, 0, 0]]),
}

# Row selector + minimum value (for a positionRel inequality objective) constraining the
# UNcentered approach/depth axis (see _GRASP_CENTER_AXES above) to be strictly positive -- i.e.
# the gripper must sit on ONE particular side of the object, not either. Without this, negDistance
# is satisfied just as well by the mirror-image solution (e.g. for 'top', touching the box's
# BOTTOM face instead of its top, gripper below the object's center) -- confirmed empirically:
# roughly 1 in 5 solves converged to a negative-depth ("from below") grasp instead of the intended
# from-above one, and since that requires the arm to reach below the table the box sits on, replay
# showed the box dipping under the floor to reach that (kinematically valid but physically
# nonsensical) grasp, briefly during the carry and again right
# before release. 0.02 is a modest clearance, not a precise value -- it only needs to be small
# enough not to conflict with the centering equality objectives above and large enough to rule out
# the near-zero boundary between the two mirror solutions.
_GRASP_DEPTH_AXIS = {
    'top': (np.array([[0, 0, 1]]), 0.02),
    'side': (np.array([[0, 1, 0]]), 0.02),
}

def rai_ik(robot, base_conf, grasp, custom_limits={}, view=False, max_attempts=6,
          arm_joints=None, gripper_frame=None, base_joints=None):
    """Fixes the base at base_conf (skipped if this robot has no base joints -- base_conf may be
    None in that case), solves for arm joint values that bring the gripper frame into a grasp of
    grasp.obj_frame_name. Returns an arm conf tuple, or None if infeasible.

    arm_joints/gripper_frame/base_joints default to robot.spec (if robot is a RaiRobot) or the
    single-mobile-robot module constants otherwise -- existing single-robot callers don't need to
    change."""
    C = _config_of(robot)
    arm_joints, gripper_frame, base_joints, _ = _spec_of(robot, arm_joints, gripper_frame, base_joints)
    if base_joints:
        set_joint_positions(C, base_joints, base_conf)
    # Restore to the config's FULL original joint set afterwards, not just this robot's own
    # (base_joints + arm_joints) -- selectJoints changes which joints the whole Config considers
    # active, and for the multi-arm scenario other robots' joints share this same Config.
    original_joints = list(C.getJointNames())
    C.selectJoints(arm_joints)
    try:
        lower, upper = get_custom_limits(C, arm_joints, custom_limits)
        for attempt in range(max_attempts):
            # enableCollisions + a soft accumulatedCollisions penalty steers the solver away from
            # self- and table-colliding solutions (a plain negDistance/scalarProduct solve regularly
            # converges to self-colliding arm configs, since nothing in the objective discourages
            # it). Soft (OT.sos) rather than hard (OT.ineq) so a tight-but-feasible grasp doesn't get
            # thrown out; get_ik_fn's check_collisions() call is the actual accept/reject gate.
            komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=True)
            komo.addControlObjective([], 0, 1e-1)
            komo.addObjective([1], ry.FS.accumulatedCollisions, [], ry.OT.sos, [3e1])
            komo.addObjective([1], ry.FS.negDistance, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.eq, [1e1])
            # Two scalarProduct constraints, not one -- a single one only pins the gripper's
            # squeeze axis to be PERPENDICULAR to one object axis, leaving a full free rotation
            # around the approach direction (confirmed empirically: five KOMO solves against the
            # same box, same grasp_type, landed on five different yaw angles). Matches
            # robotic/manipulation.py's own grasp_box() (grasp_direction='x': align=[scalarProductXY,
            # scalarProductXZ]): together, perpendicular-to-Y AND perpendicular-to-Z pins the
            # gripper's local X (its squeeze axis) to be PARALLEL to the object's own local X axis
            # (up to a 180-degree flip) -- i.e. always grasping parallel to an edge of the box,
            # never at a diagonal.
            for align_fs in _GRASP_ALIGN.get(grasp.grasp_type, _GRASP_ALIGN['side']):
                komo.addObjective([1], align_fs, [gripper_frame, grasp.obj_frame_name],
                                  ry.OT.eq, [1e1], [0.0])
            # Centers the gripper on the object's two in-plane axes (see _GRASP_CENTER_AXES
            # above) -- negDistance only requires touching the object somewhere, not centered.
            center_rows = _GRASP_CENTER_AXES.get(grasp.grasp_type, _GRASP_CENTER_AXES['side'])
            komo.addObjective([1], ry.FS.positionRel, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.eq, center_rows * 1e1)
            # Rules out the from-below mirror-image grasp on the remaining (depth) axis --
            # see _GRASP_DEPTH_AXIS above.
            depth_row, depth_min = _GRASP_DEPTH_AXIS.get(grasp.grasp_type, _GRASP_DEPTH_AXIS['side'])
            komo.addObjective([1], ry.FS.positionRel, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.ineq, depth_row * (-1e1), [depth_min])
            if attempt > 0:
                x_init = np.random.uniform(lower, upper)
                komo.initWithConstant(x_init)
            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            ret = ret.dict()
            if view:
                komo.view(True, 'IK attempt {}'.format(attempt))
            if ret['ineq'] < 1 and ret['eq'] < 1 and ret['feasible']:
                q = tuple(komo.getPath()[0])
                if all_between(lower, q, upper):
                    return q
        return None
    finally:
        C.selectJoints(original_joints)


def rai_pick_place_ik(robot, grasp, place_pose, custom_limits={}, view=False, max_attempts=6,
                      arm_joints=None, gripper_frame=None, base_joints=None):
    """Fixed-base (base_joints must be empty -- see below) joint solve for a pick-and-place
    action's TWO keyframes together, rather than as two independent rai_ik() calls. Matches the
    addModeSwitch pattern from vhartman/multirobot-pathplanning-benchmark's rai_config.py (e.g.
    compute_pick_and_place): a 2-phase KOMO problem with kOrder=1 (velocity continuity across the
    phase boundary) and an explicit ry.SY.stable mode switch making the object rigidly follow the
    gripper from phase 1 onward, so the SAME grasp offset determined at phase 1 (grasp) is what
    determines where the object ends up at phase 2 (place) -- rather than rai_ik() solving grasp
    and place independently, which can (and empirically does, for a top/side grasp's free
    rotation about the approach axis) land on two different valid touching geometries, so an arm
    that reaches BOTH confs exactly still leaves the object several cm off its intended placement
    pose. Verified in isolation against a real scenario: with the two confs applied through the
    same C.setJointState()+C.attach() replay mechanism this codebase actually uses, the object's
    final position/orientation error against place_pose was sub-millimeter (vs. ~5-14cm from two
    independent rai_ik() calls).

    place_pose: (position, quaternion) the object's frame should have once placed (e.g. from
    sample_placement()).
    Returns (grasp_conf, place_conf) arm-joint tuples, or (None, None) if infeasible.

    Only supports a fixed-mount arm (base_joints falsy): a moving base would need its own DOF in
    the 2-phase optimization (the base could legitimately reposition between pick and place),
    which this doesn't attempt -- callers with a mobile base should keep using rai_ik() twice."""
    C = _config_of(robot)
    arm_joints, gripper_frame, base_joints, _ = _spec_of(robot, arm_joints, gripper_frame, base_joints)
    if base_joints:
        raise NotImplementedError("rai_pick_place_ik() only supports a fixed-mount arm (no base_joints)")
    original_joints = list(C.getJointNames())
    C.selectJoints(arm_joints)
    try:
        lower, upper = get_custom_limits(C, arm_joints, custom_limits)
        place_pos, place_quat = place_pose
        for attempt in range(max_attempts):
            komo = ry.KOMO(C, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True)
            komo.addControlObjective([], 0, 1e-1)
            komo.addControlObjective([], 1, 1e-1)
            komo.addObjective([1, 2], ry.FS.accumulatedCollisions, [], ry.OT.sos, [3e1])

            komo.addObjective([1], ry.FS.negDistance, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.eq, [1e1])
            # See rai_ik()'s comment on _GRASP_ALIGN: two constraints, not one, so the squeeze
            # axis is pinned parallel to an object edge rather than free to rotate to any yaw.
            for align_fs in _GRASP_ALIGN.get(grasp.grasp_type, _GRASP_ALIGN['side']):
                komo.addObjective([1], align_fs, [gripper_frame, grasp.obj_frame_name],
                                  ry.OT.eq, [1e1], [0.0])
            # See rai_ik()'s comment on _GRASP_CENTER_AXES: centers the gripper on the object's
            # two in-plane axes instead of leaving it anywhere negDistance is satisfied (up to
            # the object's edge).
            center_rows = _GRASP_CENTER_AXES.get(grasp.grasp_type, _GRASP_CENTER_AXES['side'])
            komo.addObjective([1], ry.FS.positionRel, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.eq, center_rows * 1e1)
            # See rai_ik()'s comment on _GRASP_DEPTH_AXIS: rules out the from-below mirror-image
            # grasp on the remaining (depth) axis.
            depth_row, depth_min = _GRASP_DEPTH_AXIS.get(grasp.grasp_type, _GRASP_DEPTH_AXIS['side'])
            komo.addObjective([1], ry.FS.positionRel, [gripper_frame, grasp.obj_frame_name],
                              ry.OT.ineq, depth_row * (-1e1), [depth_min])

            komo.addModeSwitch([1, 2], ry.SY.stable, [gripper_frame, grasp.obj_frame_name])
            komo.addObjective([2], ry.FS.position, [grasp.obj_frame_name], ry.OT.eq, [1e1], list(place_pos))
            komo.addObjective([2], ry.FS.quaternion, [grasp.obj_frame_name], ry.OT.eq, [1e1], list(place_quat))

            if attempt > 0:
                x_init = np.random.uniform(lower, upper)
                komo.initWithConstant(x_init)
            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            ret = ret.dict()
            if view:
                komo.view(True, 'Pick-place IK attempt {}'.format(attempt))
            if ret['ineq'] < 1 and ret['eq'] < 1 and ret['feasible']:
                path = komo.getPath()
                grasp_conf, place_conf = tuple(path[0]), tuple(path[1])
                if all_between(lower, grasp_conf, upper) and all_between(lower, place_conf, upper):
                    return grasp_conf, place_conf
        return None, None
    finally:
        C.selectJoints(original_joints)


##### Grasp generation #####

def get_top_grasps(obj_frame_name, carry=PANDA_CARRY_CONF):
    return [Grasp('top', obj_frame_name, carry=carry)]


def get_side_grasps(obj_frame_name, carry=PANDA_CARRY_CONF):
    return [Grasp('side', obj_frame_name, carry=carry)]


##### Base pose sampling (IR / reachability) #####

def uniform_pose_generator(C, target_xy, min_radius=0.55, max_radius=0.7):
    while True:
        radius = random.uniform(min_radius, max_radius)
        theta = random.uniform(-math.pi, math.pi)
        x = target_xy[0] + radius * math.cos(theta)
        y = target_xy[1] + radius * math.sin(theta)
        facing = math.atan2(target_xy[1] - y, target_xy[0] - x)
        yield (x, y, facing)


##### Frame helpers #####

def _gripper_finger_names(gripper_frame):
    """'l_gripper'/'r_gripper' -> ['l_panda_finger_joint1', 'l_panda_finger_joint2']."""
    prefix = gripper_frame[:-len('gripper')] if gripper_frame.endswith('gripper') else gripper_frame
    return [f'{prefix}panda_finger_joint{i}' for i in (1, 2)]


_finger_open_local_offset_cache = {}


def _finger_open_local_offset(C, gripper_frame, name):
    """The finger's natural 'fully open' offset from the gripper frame, expressed in the
    gripper's own local coordinates -- captured once, on first use (from the scenario file's own
    untouched pose, since this must run before anything moves the arm), and cached that way
    rather than as a raw world position so it stays valid as the arm/gripper moves later in the
    replay (a raw world offset, captured once at the start, goes stale and points nowhere near
    the gripper the moment the arm leaves its initial configuration)."""
    key = (id(C), name)
    offset = _finger_open_local_offset_cache.get(key)
    if offset is None:
        grip = C.getFrame(gripper_frame)
        grip_pos = np.asarray(grip.getPosition())
        R = _quat_to_rotation_matrix(grip.getQuaternion())
        open_pos = np.asarray(C.getFrame(name).getPosition())
        offset = R.T @ (open_pos - grip_pos)
        _finger_open_local_offset_cache[key] = offset
    return offset


def set_gripper_fingers(C, gripper_frame, opening=None):
    """Moves a gripper's finger frames directly to a world position interpolated between the
    scenario file's own natural 'fully open' pose (captured once, on first use, before any
    animation touches them) and the gripper's own reference point. `opening` is the desired
    half-gap in meters (None, or >= the natural half-gap, means fully open; 0 means fully
    closed, touching at the gripper's own point).

    The Panda's finger joints are joint_active: False by default in panda.g/pandasTable.g and
    mimic-linked to each other in the .g file; reactivating them as independent transY DOFs via
    frame.setJoint() and driving them through C.setJointState() was tried first but doesn't
    work here -- both fingers collapsed onto the identical world position for every input
    (confirmed empirically: (0,0), (0.02,0.02) and (0.04,0.04) all produced zero gap between
    them), because RAI keeps honoring the original mimic linkage regardless of the explicit
    per-joint values passed in. This bypasses the joint/mimic system entirely and just
    repositions the frames directly -- purely visual, same mechanism already relied on
    elsewhere in this module for repositioning objects after C.attach()."""
    C = _config_of(C)
    grip = C.getFrame(gripper_frame)
    grip_pos = np.asarray(grip.getPosition())
    R = _quat_to_rotation_matrix(grip.getQuaternion())
    for name in _gripper_finger_names(gripper_frame):
        f = C.getFrame(name)
        if f is None:
            continue
        local_offset = _finger_open_local_offset(C, gripper_frame, name)
        natural = np.linalg.norm(local_offset)
        if natural < 1e-9:
            continue
        target = natural if opening is None else min(max(opening, 0.0), natural)
        f.setPosition(grip_pos + R @ (local_offset / natural * target))


def gripper_finger_axis(C, gripper_frame):
    """World-frame unit direction the gripper's fingers separate along, derived from finger1's
    natural local offset (the same cached offset set_gripper_fingers uses) rotated by the
    gripper's current orientation. Used to compute the correct finger-closing width for a held
    object via box_support_distance -- the object's footprint across this specific direction,
    not just its narrowest local dimension, since a top/side grasp permits free rotation around
    the approach axis and the box's local X/Y sizes don't track that rotation."""
    C = _config_of(C)
    name = _gripper_finger_names(gripper_frame)[0]
    local_offset = _finger_open_local_offset(C, gripper_frame, name)
    norm = np.linalg.norm(local_offset)
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    grip = C.getFrame(gripper_frame)
    R = _quat_to_rotation_matrix(grip.getQuaternion())
    return R @ (local_offset / norm)


def _quat_to_rotation_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_support_distance(direction_world, quat, size):
    """Distance from a box's center to its own surface along a world-frame direction, for a box
    with the given orientation and full extents `size`. Standard box-support formula: rotate the
    direction into the box's own local frame, then the exit distance for an axis-aligned box of
    half-extents h along a unit direction d is 1 / max_i(|d_i| / h_i). Verified empirically
    against real KOMO negDistance=0 grasp solves (mm_drrt_ik-style top grasps): this matches the
    solved gripper-to-object distance to within ~1mm, with no extra clearance term needed."""
    direction_world = np.asarray(direction_world, dtype=float)
    norm = np.linalg.norm(direction_world)
    if norm < 1e-9:
        return float(np.max(size) / 2.0)
    direction_world = direction_world / norm
    R = _quat_to_rotation_matrix(quat)
    local_dir = R.T @ direction_world
    half = np.asarray(size) / 2.0
    ratios = np.abs(local_dir) / np.maximum(half, 1e-9)
    denom = np.max(ratios)
    return 1.0 / denom if denom > 1e-9 else float(np.max(half))


def remove_frame(C, frame_name):
    _config_of(C).delFrame(frame_name)


def get_frame_position(C, frame_name):
    return tuple(_config_of(C).getFrame(frame_name).getPosition())


def is_placement(C, obj_frame_name, surface_frame_name, epsilon=0.02):
    C = _config_of(C)
    obj = C.getFrame(obj_frame_name)
    surface = C.getFrame(surface_frame_name)
    obj_pos = np.asarray(obj.getPosition())
    surf_pos = np.asarray(surface.getPosition())
    obj_size = np.asarray(obj.getSize()[:3])
    surf_size = np.asarray(surface.getSize()[:3])
    surface_top_z = surf_pos[2] + surf_size[2] / 2.0
    obj_bottom_z = obj_pos[2] - obj_size[2] / 2.0
    within_xy = (abs(obj_pos[0] - surf_pos[0]) <= surf_size[0] / 2.0 + obj_size[0] / 2.0 and
                abs(obj_pos[1] - surf_pos[1]) <= surf_size[1] / 2.0 + obj_size[1] / 2.0)
    return within_xy and abs(obj_bottom_z - surface_top_z) <= epsilon


def sample_placement(C, obj_frame_name, surface_frame_name, margin=0.02):
    C = _config_of(C)
    surface = C.getFrame(surface_frame_name)
    surf_pos = np.asarray(surface.getPosition())
    surf_size = np.asarray(surface.getSize()[:3])
    obj = C.getFrame(obj_frame_name)
    obj_size = np.asarray(obj.getSize()[:3])
    half_x = max(surf_size[0] / 2.0 - obj_size[0] / 2.0 - margin, 0.0)
    half_y = max(surf_size[1] / 2.0 - obj_size[1] / 2.0 - margin, 0.0)
    if half_x <= 0 or half_y <= 0:
        return None
    x = surf_pos[0] + random.uniform(-half_x, half_x)
    y = surf_pos[1] + random.uniform(-half_y, half_y)
    z = surf_pos[2] + surf_size[2] / 2.0 + obj_size[2] / 2.0
    return ((x, y, z), (1.0, 0.0, 0.0, 0.0))


##### Save / restore world state #####

class ConfigSaver(object):
    def __init__(self, C):
        self.C = _config_of(C)
        C = self.C
        self.joint_names = list(C.getJointNames())
        self.joint_state = np.asarray(C.getJointState()).copy()
        self.frame_names = [f.name for f in C.getFrames()]
        self.frame_poses = {
            name: (tuple(C.getFrame(name).getPosition()), tuple(C.getFrame(name).getQuaternion()))
            for name in self.frame_names
        }

    def restore(self):
        for name, (point, quat) in self.frame_poses.items():
            f = self.C.getFrame(name)
            if f is None:
                continue
            f.setPosition(point)
            f.setQuaternion(quat)
        self.C.setJointState(self.joint_state, self.joint_names)


def save_world(C):
    return ConfigSaver(C)


def restore_world(saver):
    saver.restore()


##### Misc, generic helpers (no RAI dependency; kept local so this module doesn't need pybullet) #####

def elapsed_time(start_time):
    return time.time() - start_time


def irange(start, end=None, step=1):
    if end is None:
        end, start = start, 0
    n = start
    while n < end:
        yield n
        n += step


def all_close(a, b, atol=1e-6, rtol=0.):
    assert len(a) == len(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        if not all_close(new_path[-1], conf, atol=tolerance):
            new_path.append(conf)
    return new_path


def get_unit_vector(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
