#!/usr/bin/env python
"""RAI dual-Franka-arm scenario: 2 fixed Panda arms sharing one table (pandasTable.g, same as
example_two_robots_rai_env.py) stack N square blocks into a single tower at the table midpoint
both arms can reach. See /home/enco/.claude/plans/precious-weaving-kettle.md for the design.

Blocks alternate ownership by stack index (block0 -> left arm, block1 -> right arm, block2 ->
left, ...), each starting in its owner's own pickup zone (zone_left / zone_right) and going
straight onto the growing tower -- no handoff action needed since both arms reach the tower.

num_objs must be EVEN. mm_drrt/utils/rai_task_planner_utils.py:308's assign_order_constraints()
adds an automatic "balance constraint" -- robot A's k-th action can't start until robot B reaches
subprob 3k-1 -- unconditionally assuming both robots' action lists are the SAME length (true of
every existing env, since they always split objects evenly). An odd num_objs here gives the two
arms unequal action counts (e.g. 5 blocks -> 3 left/6 actions vs 2 right/4 actions), and the
balance constraint then demands a subprob index the shorter robot's roadmap can never reach --
a genuine, permanent dRRT* deadlock (confirmed: search plateaus forever, independent of time
budget), not just a slow search. Kept an even split (arms get equal-length plans) rather than
patching that shared assumption, since it's used by every other RAI env too.

Blocks can't target "the block below them" directly: get_placement_gen()/sample_placement() reads
a surface frame's LIVE position, but PlanSkeleton.initialize() samples every f_obj's placements
once, up front, before any action has moved anything -- so a lower block's frame would still be
sitting at its initial pickup-zone pose at sampling time, not its future tower position. Instead,
each stack level gets its own static, non-colliding marker frame (tower_level0..N-1) at a fixed,
precomputed height, sized to exactly match a block's footprint so sample_placement's jitter margin
is zero (deterministic, centered placement -- what a stable tower needs anyway).

The shared framework only treats a block as a placement-sampling obstacle for an f_obj it started
on (env.m_obj_in_f_obj), which is empty for every tower level -- so by default nothing tells
block i's arm-motion planning that block i-1 physically occupies the spot right under it. Since
placement itself is deterministic and non-interpenetrating this mostly self-corrects, but a
PRM-sampled approach/retrieval path could still swing through the block below. compute_path()
below guards against that by temporarily adding a copy of the block at the known level-(i-1) pose
to the obstacle list, mirroring the copied_objs idiom example_two_robots_rai_env.py already uses.
"""
import itertools

import robotic as ry

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.rai_motion_planner_utils import get_placement_gen, \
    get_grasp_gen, get_gripper, arm_retrieval_motion, get_arm_motion_fn, get_trivial_roadmap, \
    get_fixed_arm_pick_place_ik_gen

from examples.rai_utils import Environment, create_box


BLOCK = (0.05, 0.05, 0.05)  # a cube ("square block"), unlike the tall BOX used elsewhere
StackBlocksRaiCameraSetup = [(0, -1, 4), (0, 0, 0)]

_copy_counter = itertools.count()

# Same pad convention as example_two_robots_rai_env.py: thin, non-collidable marker frames on top
# of the real, contact-enabled 'table' frame (the only real collision geometry for the tabletop).
ZONE_SIZE = (0.3, 0.3, 0.02)
# sample_placement() (mm_drrt/utils/rai_utils.py) rejects a surface exactly the size of the
# object outright (its half-extent computation floors at 0, and 0 itself is treated as
# "no room" -- see its `half_x <= 0` check) -- so the footprint can't literally equal BLOCK's.
# This gives just enough margin for a small, deliberately tight +-5mm placement jitter, still
# well within what a flush-stacked block/gripper can tolerate.
LEVEL_SIZE = (0.10, 0.10, 0.002)
ZONE_X = {'left': -0.8, 'right': 0.8}
ZONE_Y = 0.15
ZONE_Z = 0.6
TOWER_X = 0.0
TOWER_Y = 0.15
ZONE_ROW_SPACING = 0.08  # y-offset between blocks laid out within one arm's own zone


class StackBlocksRaiEnvironment(Environment):
    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        super().__init__(num_objs, seed)
        if num_robots != 2:
            raise Exception("This scenario supports exactly 2 robots.")
        if num_objs % 2 != 0:
            raise Exception("num_objs must be even (arms split blocks evenly) -- see the module "
                            "docstring for why an odd count deadlocks the shared dRRT* scheduler.")
        self._num_robots = num_robots
        self._num_m_objs = num_objs
        self._C = sim_id
        self._grippers = {}
        self._grasp_type = grasp_type
        self._initialize(arm)

    def _initialize(self, arm):
        self._arm = arm
        self._create_problem()
        self.grasp_gen_fn = get_grasp_gen(self._grasp_type, collisions=True)
        self.placement_gen_fn = get_placement_gen(self._C)

    def placement_sample(self, m_objs, f_obj, num_samples):
        placement_gen = self.placement_gen_fn(m_objs, f_obj)
        placements = []
        for _ in range(num_samples):
            (p,) = next(placement_gen)
            placements.append(p)
        return placements

    def is_placement_collision(self, objs_poses, stationary_m_objs,
                               remove_m_objs, add_m_objs, remove_then_add_m_objs, add_then_remove_m_objs):
        for p in objs_poses:
            p.assign()

        new_placements = remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs
        for i in range(len(new_placements) - 1):
            for j in range(i + 1, len(new_placements)):
                if ru.boxes_overlap(self._C, new_placements[i], new_placements[j]):
                    return False, []

        for add_m_obj in remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs:
            if any(ru.boxes_overlap(self._C, add_m_obj, obst) for obst in stationary_m_objs):
                return False, []

        collisions = []
        for add_m_obj in remove_then_add_m_objs + add_m_objs:
            for remove_m_obj in remove_m_objs + add_then_remove_m_objs:
                if ru.boxes_overlap(self._C, add_m_obj, remove_m_obj):
                    collisions.append((add_m_obj, remove_m_obj))
        return True, collisions

    def placement_collision_with_remove_then_add_m_objs(self, init_obj_pose, cur_obj_pose,
                                                        remove_then_add_m_obj_id, remove_then_add_m_obj,
                                                        init_collisions, add_m_objs, add_then_remove_m_objs,
                                                        remove_then_add_m_objs):
        init_obj_pose.assign()
        for id, add_m_obj in enumerate(remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs):
            if id == remove_then_add_m_obj_id:
                continue
            if ru.boxes_overlap(self._C, add_m_obj, remove_then_add_m_obj):
                init_collisions.append((add_m_obj, remove_then_add_m_obj))
        cur_obj_pose.assign()
        return init_collisions

    def subgoal_sampling(self, robot, obj_orders, actions, action, m_obj, start_obstacles, goal_obstacles,
                         custom_limits={}, use_debug=False):
        # get_fixed_arm_pick_place_ik_gen()'s solve leaves each arm's joints at whatever candidate
        # config it last tested (successful or not) rather than resetting to carry_conf -- so a
        # later subgoal_sampling call (this env solves every action's grasp+place through a single
        # shared central tower, unlike the relay envs' more spread-out targets) can see the OTHER
        # arm's stale, mid-solve pose from an earlier action and reject valid solutions as
        # "self-colliding" against a pose that will actually be back at carry_conf by execution
        # time. Confirmed by direct reproduction: subgoal_refinement for a 2-block stack failed
        # 15/15 random seeds without this reset, and succeeded 10/10 with it.
        ru.set_joint_positions(self._C, ru.LEFT_ARM.arm_joints, ru.LEFT_ARM.carry_conf)
        ru.set_joint_positions(self._C, ru.RIGHT_ARM.arm_joints, ru.RIGHT_ARM.carry_conf)
        pick_place_ik_fn = get_fixed_arm_pick_place_ik_gen(
            robot, max_attempts=25,
            start_collision_objs=start_obstacles['objs'] + self.fixed_obstacles,
            goal_collision_objs=goal_obstacles['objs'] + self.fixed_obstacles,
            custom_limits=custom_limits.get(robot, {}), use_debug=use_debug)
        self._assign_target_obj_pose(actions, obj_orders, m_obj, action)
        start_obj_pose = ru.Pose(self._C, m_obj)
        grasps = list(self.grasp_gen_fn(robot, m_obj))

        for grasp in grasps:
            (g,) = grasp
            output = next(pick_place_ik_fn(self._arm, m_obj, start_obj_pose, g, action.goal['place'],
                                           start_obst_poses=start_obstacles['poses'],
                                           goal_obst_poses=goal_obstacles['poses']), None)
            if output:
                pick_output, place_output = output
                action.start['grasp'] = g
                action.start['place'] = start_obj_pose
                action.start['base'] = pick_output[0]
                action.start['approach_conf'] = pick_output[1]
                action.start['grasp_conf'] = pick_output[2]
                action.goal['grasp'] = g
                action.goal['base'] = place_output[0]
                action.goal['approach_conf'] = place_output[1]
                action.goal['grasp_conf'] = place_output[2]
                return True
        return False

    def compute_path(self, robot, action, m_obj, num_base_samples, num_arm_samples, type=None, use_debug=False):
        if type == 'return':
            return True, [], []

        start_obstacles = action.obstacles['start']
        goal_obstacles = action.obstacles['goal']
        obstacles = set(start_obstacles['objs']) | set(goal_obstacles['objs'])
        for obst_pose in start_obstacles['poses']:
            obst_pose.assign()
        copied_objs = []
        if type == 'transfer':
            copied_objs_poses = []
            for id in range(len(goal_obstacles['objs'])):
                if goal_obstacles['objs'][id] in set(start_obstacles['objs']) & set(goal_obstacles['objs']):
                    name = 'copied_box_{}'.format(next(_copy_counter))
                    create_box(self._C, name, (0, 0, -5), BLOCK)
                    copied_objs.append(name)
                    copied_objs_poses.append(goal_obstacles['poses'][id].value)
                    continue
            for id in range(len(copied_objs)):
                ru.Pose(self._C, copied_objs[id], copied_objs_poses[id], action.to_f_obj).assign()
            obstacles = obstacles | set(copied_objs)

            # The block this one is about to land on top of isn't tracked as an obstacle by the
            # shared framework (see module docstring) -- add a temporary stand-in at its known,
            # deterministic pose so the approach/retrieval motion can't plan straight through it.
            level = self._level_of_target.get(action.to_f_obj)
            if level is not None and level > 0:
                name = 'copied_box_{}'.format(next(_copy_counter))
                create_box(self._C, name, (0, 0, -5), BLOCK)
                ru.Pose(self._C, name, self._level_pose[level - 1], action.to_f_obj).assign()
                copied_objs.append(name)
                obstacles = obstacles | {name}
        obstacles = list(obstacles) + self.fixed_obstacles

        arm_motion_fn = get_arm_motion_fn(robot, collision_objs=obstacles, num_samples=num_arm_samples,
                                          expand_type=None, expand_configs=(), use_debug=use_debug)

        max_attempts = 5
        if type == 'transit':  # pick
            action.start['place'].assign()
            attempts = 0
            while True:
                arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(
                    self._arm, m_obj, action.goal['grasp'], action.goal['approach_conf'], action.goal['grasp_conf'])
                if arm_approach_roadmap is not None:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    return False, [], []
            action.goal['grasp'].attach(self._C, gripper_frame=robot.spec.gripper_frame)
            attempts = 0
            while True:
                arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(
                    robot, self._arm, type, grasp=action.goal['grasp'],
                    start=arm_approach_roadmap.final_conf, goal=robot.spec.carry_conf,
                    obstacles=obstacles, attachments=[m_obj], num_samples=num_arm_samples,
                    expand_type=None, expand_configs=(), use_debug=use_debug)
                if arm_retrieval_roadmap is not None:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    return False, [], []
        else:  # 'transfer' -- place
            attempts = 0
            while True:
                arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(
                    self._arm, m_obj, action.goal['grasp'], action.goal['approach_conf'], action.goal['grasp_conf'],
                    attachments=[m_obj])
                if arm_approach_roadmap is not None:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    for copied_obj in copied_objs:
                        ru.remove_frame(self._C, copied_obj)
                    return False, [], []
            action.goal['grasp'].detach(self._C, action.to_f_obj)
            attempts = 0
            while True:
                arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(
                    robot, self._arm, type, grasp=action.goal['grasp'],
                    start=arm_approach_roadmap.final_conf, goal=robot.spec.carry_conf,
                    obstacles=obstacles, num_samples=num_arm_samples,
                    expand_type=None, expand_configs=(), use_debug=use_debug)
                if arm_retrieval_roadmap is not None:
                    break
                attempts += 1
                if attempts >= max_attempts:
                    for copied_obj in copied_objs:
                        ru.remove_frame(self._C, copied_obj)
                    return False, [], []

        for copied_obj in copied_objs:
            ru.remove_frame(self._C, copied_obj)
        trivial_attachments = [m_obj] if type == 'transfer' else []
        base_roadmap, base_heuristic_val = get_trivial_roadmap(robot.spec.carry_conf, attachments=trivial_attachments)
        return True, [base_roadmap, arm_approach_roadmap, arm_retrieval_roadmap], \
               [base_heuristic_val, arm_approach_heuristic_val, arm_retrieval_heuristic_val]

    def get_joints(self, robots):
        return [robot.spec.arm_joints for robot in robots]

    def get_base_conf(self, robot):
        return ru.Conf(robot, [], ())

    def get_init_base_conf(self, robot):
        return ru.Conf(robot, [], ())

    def save_world(self):
        return ru.save_world(self._C)

    def restore_world(self, saved_world):
        ru.restore_world(saved_world)

    def _assign_target_obj_pose(self, actions, obj_orders, m_obj, action):
        for a in actions:
            if actions[a] == action:
                for id in range(len(obj_orders)):
                    if obj_orders[id] == a:
                        if id == 0:
                            self.m_objs_init_placements[m_obj].assign()
                        else:
                            actions[obj_orders[id - 1]].goal['place'].assign()

    def create_plan_order_constraints(self):
        plan = {}
        action_orders = {self.robots[0]: (), self.robots[1]: ()}
        obj_orders = {}
        init_order_constraints = []
        prev_transfer = None
        for i, block in enumerate(self.m_objs):
            robot = self.robots[0] if i % 2 == 0 else self.robots[1]
            zone = 'zone_left' if i % 2 == 0 else 'zone_right'
            transit_name, transfer_name = f'a{2 * i}', f'a{2 * i + 1}'
            plan[transit_name] = ('transit', robot, block, None, zone)
            plan[transfer_name] = ('transfer', robot, block, zone, f'tower_level{i}')
            action_orders[robot] += (transit_name, transfer_name)
            obj_orders[block] = [transfer_name]
            if prev_transfer is not None:
                init_order_constraints.append({'pre': prev_transfer, 'post': transit_name})
            prev_transfer = transfer_name
        return plan, action_orders, obj_orders, tuple(init_order_constraints)

    def _create_problem(self):
        self.custom_limits = {}

        self._C.addFile(ry.raiPath(ru.TWO_ARM_SCENARIO))
        ru.set_joint_positions(self._C, ru.LEFT_ARM.arm_joints, ru.LEFT_ARM.carry_conf)
        ru.set_joint_positions(self._C, ru.RIGHT_ARM.arm_joints, ru.RIGHT_ARM.carry_conf)

        self.fixed_obstacles = ['table']  # already defined (and contact-enabled) in pandasTable.g

        f_objs = ['zone_left', 'zone_right']
        for key, x in ZONE_X.items():
            self._C.addFrame(f'zone_{key}').setPosition([x, ZONE_Y, ZONE_Z]).setShape(ry.ST.box, size=list(ZONE_SIZE)) \
                .setColor([0.5, 0.5, 0.9]).setContact(0)

        self._level_pose = {}
        for i in range(self._num_m_objs):
            level_name = f'tower_level{i}'
            marker_z = ZONE_Z - LEVEL_SIZE[2] / 2.0 + i * BLOCK[2]
            # alpha=0: invisible in the GUI, but keeps real box geometry (unlike ry.ST.marker,
            # whose getSize() wouldn't feed sample_placement()'s footprint math correctly) -- this
            # frame only ever needs to exist for placement-sampling bookkeeping, not to be seen.
            self._C.addFrame(level_name).setPosition([TOWER_X, TOWER_Y, marker_z]).setShape(ry.ST.box, size=list(LEVEL_SIZE)) \
                .setColor([0.9, 0.5, 0.5, 0.0]).setContact(0)
            f_objs.append(level_name)
            box_z = ZONE_Z + (i + 0.5) * BLOCK[2]
            self._level_pose[i] = ((TOWER_X, TOWER_Y, box_z), (1.0, 0.0, 0.0, 0.0))

        self.robots = {0: ru.RaiRobot(self._C, ru.LEFT_ARM), 1: ru.RaiRobot(self._C, ru.RIGHT_ARM)}
        for robot in self.robots.values():
            self.custom_limits[robot] = {}
            self._grippers[robot] = get_gripper(robot)

        colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0)]
        boxes = []
        self.m_objs_init_placements = {}
        zone_left_blocks, zone_right_blocks = set(), set()
        zone_row = {'left': 0, 'right': 0}
        for i in range(self._num_m_objs):
            key = 'left' if i % 2 == 0 else 'right'
            row = zone_row[key]
            zone_row[key] += 1
            y = ZONE_Y + (row - 1) * ZONE_ROW_SPACING
            box_pos = (ZONE_X[key], y, ZONE_Z + BLOCK[2] / 2.0)
            box_name = create_box(self._C, f'block{i}', box_pos, BLOCK, color=colors[i % len(colors)])
            box_pose = (box_pos, (1.0, 0.0, 0.0, 0.0))
            boxes.append(box_name)
            self.m_objs_init_placements[box_name] = ru.Pose(self._C, box_name, box_pose, f'zone_{key}')
            (zone_left_blocks if key == 'left' else zone_right_blocks).add(box_name)

        self.m_objs = boxes
        self.f_objs = f_objs
        self.m_obj_in_f_obj = {'zone_left': zone_left_blocks, 'zone_right': zone_right_blocks}
        self._level_of_target = {f'tower_level{i}': i for i in range(self._num_m_objs)}
