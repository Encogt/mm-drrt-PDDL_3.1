#!/usr/bin/env python
"""RAI port of examples/envs/example_two_robots_env.py's 1-object relay scenario, but for two
FIXED Franka arms sharing one table (RAI's 'scenarios/pandasTable.g') instead of two mobile
PR2s driving between three separate tables. See /home/enco/.claude/plans/deep-wobbling-sonnet.md.

Layout (top view, y=0.15 line on the shared table): table_start (x=-0.8, l_panda-only reach) --
table_mid (x=0.0, both arms reach) -- table_end (x=+0.8, r_panda-only reach). l_panda picks the
box from table_start and places it at table_mid; r_panda picks it up from there and places it at
table_end.

No base motion exists for either arm (pandasTable.g mounts both rigidly) -- compute_path plans
arm motion directly with expand_type=None (no base dimension to concatenate; see
mm_drrt/planner/prm.py's prm(), which already treats expand_type=None as "no expansion"), but
still contributes a *placeholder* base roadmap per action (get_trivial_roadmap()) rather than
dropping it: the multi-robot dRRT* machinery (task_planner_utils.py's assign_order_constraints,
subprob_id indexing) hardcodes "3 roadmap segments per action" to match the mobile-base
scenario's [base, arm-approach, arm-retrieval] shape, and returning only 2 misaligns that
indexing between robots.
"""
import itertools

import robotic as ry

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.rai_motion_planner_utils import get_fixed_arm_ik_ir_gen, get_placement_gen, \
    get_grasp_gen, get_gripper, arm_retrieval_motion, get_arm_motion_fn, get_trivial_roadmap

from examples.rai_utils import Environment, create_box


BOX = (.07, .05, .15)
ExampleTwoRobotsRaiCameraSetup = [(0, -1, 4), (0, 0, 0)]

_box_counter = itertools.count()

# Table top is at z=0.6 (see pandasTable.g); surfaces are thin, non-collidable marker pads on it
# purely for placement-sampling/bookkeeping -- the table itself (already in pandasTable.g,
# contact-enabled) is the only real collision geometry for the tabletop.
SURFACE_SIZE = (0.3, 0.3, 0.02)
SURFACE_Y = 0.15
SURFACE_Z = 0.6
SURFACE_X = {'start': -0.8, 'mid': 0.0, 'end': 0.8}


class ExampleTwoRobotsRaiEnvironment(Environment):
    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        super().__init__(num_objs, seed)
        if num_robots != 2 or num_objs != 1:
            raise Exception("This scenario supports exactly 2 robots and 1 object (relay).")
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

        for i in range(len(remove_then_add_m_objs + add_m_objs) - 1):
            for j in range(i + 1, len(remove_then_add_m_objs + add_m_objs)):
                if ru.pairwise_collision(self._C, (remove_then_add_m_objs + add_m_objs)[i],
                                        (remove_then_add_m_objs + add_m_objs)[j]):
                    return False, []

        for add_m_obj in remove_then_add_m_objs + add_then_remove_m_objs + add_m_objs:
            if any(ru.pairwise_collision(self._C, add_m_obj, obst) for obst in stationary_m_objs):
                return False, []

        collisions = []
        for add_m_obj in remove_then_add_m_objs + add_m_objs:
            for remove_m_obj in remove_m_objs + add_then_remove_m_objs:
                if ru.pairwise_collision(self._C, add_m_obj, remove_m_obj):
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
            if ru.pairwise_collision(self._C, add_m_obj, remove_then_add_m_obj):
                init_collisions.append((add_m_obj, remove_then_add_m_obj))
        cur_obj_pose.assign()
        return init_collisions

    def subgoal_sampling(self, robot, obj_orders, actions, action, m_obj, start_obstacles, goal_obstacles,
                         custom_limits={}, use_debug=False):
        start_ik_ir_fn = get_fixed_arm_ik_ir_gen(robot, max_attempts=25,
                                                 collision_objs=start_obstacles['objs'] + self.fixed_obstacles,
                                                 custom_limits=custom_limits.get(robot, {}), use_debug=use_debug)
        goal_ik_ir_fn = get_fixed_arm_ik_ir_gen(robot, max_attempts=25,
                                                collision_objs=goal_obstacles['objs'] + self.fixed_obstacles,
                                                custom_limits=custom_limits.get(robot, {}), use_debug=use_debug)
        self._assign_target_obj_pose(actions, obj_orders, m_obj, action)
        start_obj_pose = ru.Pose(self._C, m_obj)
        grasps = list(self.grasp_gen_fn(robot, m_obj))

        for grasp in grasps:
            (g,) = grasp
            for obst_pose in start_obstacles['poses']:
                obst_pose.assign()
            pick_output = next(start_ik_ir_fn(self._arm, m_obj, start_obj_pose, g), None)
            if pick_output:
                if pick_output[-1]:
                    for obst_pose in goal_obstacles['poses']:
                        obst_pose.assign()
                    place_output = next(goal_ik_ir_fn(self._arm, m_obj, action.goal['place'], g), None)
                    if place_output:
                        if place_output[-1]:
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
            # Never reached by this env's plan (create_plan_order_constraints has no 'return'
            # action -- see individual_path_computation in task_planner_utils.py), and there's no
            # base to return anyway. Kept only for interface parity with other envs.
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
                    name = 'copied_box_{}'.format(next(_box_counter))
                    create_box(self._C, name, (0, 0, -5), BOX)
                    copied_objs.append(name)
                    copied_objs_poses.append(goal_obstacles['poses'][id].value)
                    continue
            for id in range(len(copied_objs)):
                ru.Pose(self._C, copied_objs[id], copied_objs_poses[id], action.to_f_obj).assign()
            obstacles = obstacles | set(copied_objs)
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
            # Object has been placed at its goal pose; release it (see the single-robot RAI env's
            # compute_path for why this explicit detach is needed -- RAI's attach() is a
            # persistent kinematic-tree reparent, unlike pybullet's per-sample Attachment.assign()).
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
        # This action's leading placeholder represents "idling before this action starts", not
        # "before picking up anything ever" -- a transfer always starts right after this same
        # robot's own transit just grasped, so it must report holding m_obj (see
        # get_trivial_roadmap()'s docstring for why get_subattachments() needs this to be
        # accurate, not just cosmetically close).
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
        # Relay: robot0 (left arm) picks from table_start -> table_mid, robot1 (right arm) picks
        # from table_mid -> table_end. Mirrors example_two_robots_env.py's 1-object relay mode.
        plan = {
            'a0': ('transit',  self.robots[0], self.m_objs[0], None,           self.f_objs[0]),
            'a1': ('transfer', self.robots[0], self.m_objs[0], self.f_objs[0], self.f_objs[1]),
            'a2': ('transit',  self.robots[1], self.m_objs[0], None,           self.f_objs[1]),
            'a3': ('transfer', self.robots[1], self.m_objs[0], self.f_objs[1], self.f_objs[2]),
        }
        action_orders = {self.robots[0]: ('a0', 'a1'),
                         self.robots[1]: ('a2', 'a3')}
        obj_orders = {self.m_objs[0]: ['a1', 'a3']}
        init_order_constraints = ({'pre': 'a1', 'post': 'a2'},)
        return plan, action_orders, obj_orders, init_order_constraints

    def create_pddl_problem(self):
        """PDDL 2.1 problem for the 1-object / 2-robot relay scenario (same shape as
        example_two_robots_env.py's, robot-can-reach constrained to each arm's own zone)."""
        objects = {
            'robot':       [self.robots[0], self.robots[1]],
            'movable-obj': [self.m_objs[0]],
            'fixed-obj':   [self.f_objs[0], self.f_objs[1], self.f_objs[2]],
        }
        init_state = [
            ('robot-free',         self.robots[0]),
            ('robot-free',         self.robots[1]),
            ('robot-at-base',      self.robots[0]),
            ('robot-at-base',      self.robots[1]),
            ('obj-location',       self.m_objs[0], self.f_objs[0]),
            ('obj-clear',          self.m_objs[0]),
            ('surface-accessible', self.f_objs[0]),
            ('surface-accessible', self.f_objs[1]),
            ('surface-accessible', self.f_objs[2]),
            ('robot-can-reach',    self.robots[0], self.f_objs[0]),
            ('robot-can-reach',    self.robots[0], self.f_objs[1]),
            ('robot-can-reach',    self.robots[1], self.f_objs[1]),
            ('robot-can-reach',    self.robots[1], self.f_objs[2]),
        ]
        goal_state = [
            ('obj-location', self.m_objs[0], self.f_objs[2]),
            ('robot-free',   self.robots[0]),
            ('robot-free',   self.robots[1]),
        ]
        return objects, init_state, goal_state

    def _create_problem(self):
        self.custom_limits = {}

        self._C.addFile(ry.raiPath(ru.TWO_ARM_SCENARIO))
        ru.set_joint_positions(self._C, ru.LEFT_ARM.arm_joints, ru.LEFT_ARM.carry_conf)
        ru.set_joint_positions(self._C, ru.RIGHT_ARM.arm_joints, ru.RIGHT_ARM.carry_conf)

        self.fixed_obstacles = ['table']  # already defined (and contact-enabled) in pandasTable.g

        f_objs = []
        for key, name in [('start', 'table_start'), ('mid', 'table_mid'), ('end', 'table_end')]:
            pos = (SURFACE_X[key], SURFACE_Y, SURFACE_Z)
            self._C.addFrame(name).setPosition(list(pos)).setShape(ry.ST.box, size=list(SURFACE_SIZE)) \
                .setColor([0.5, 0.5, 0.9]).setContact(0)
            f_objs.append(name)

        box_z = SURFACE_Z + SURFACE_SIZE[2] / 2.0 + BOX[2] / 2.0
        box_pos = (SURFACE_X['start'], SURFACE_Y, box_z)
        box_name = create_box(self._C, 'box0', box_pos, BOX)
        box_pose = (box_pos, (1.0, 0.0, 0.0, 0.0))

        self.robots = {0: ru.RaiRobot(self._C, ru.LEFT_ARM), 1: ru.RaiRobot(self._C, ru.RIGHT_ARM)}
        for robot in self.robots.values():
            self.custom_limits[robot] = {}
            self._grippers[robot] = get_gripper(robot)

        self.m_objs = [box_name]
        self.f_objs = f_objs
        self.m_objs_init_placements = {box_name: ru.Pose(self._C, box_name, box_pose, f_objs[0])}
        self.m_obj_in_f_obj = {f_objs[0]: {box_name}}
