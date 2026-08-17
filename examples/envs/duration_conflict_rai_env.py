#!/usr/bin/env python
"""RAI dual-Franka-arm scenario built to demonstrate what a WRONG PDDL action :duration causes.

Same 'scenarios/pandasTable.g' dual-arm table as example_two_robots_rai_env.py, but with a
layout that isolates a specific gap: 2 blocks, one per robot, each starting in its own zone
(zone_left / zone_right) and independently delivered onto ONE shared drop_pad both arms can
reach. Unlike example_two_robots_rai_env.py's relay/handoff scenarios, the two robots' action
chains share NO object -- so there is no same-object causal dependency linking them.

mm_drrt.utils.pddl_parser.extract_sequential_constraints only derives cross-robot ordering from
same-object handoffs (robot A transfers object O, robot B later transits the SAME O). It has no
way to infer that robot0's transfer-to-drop_pad and robot1's transfer-to-drop_pad need to be kept
apart, because no object links them. That's true regardless of the PDDL :duration value used --
create_pddl_problem() below reuses the stock mm_drrt_manipulation domain unmodified and is
solvable by Tamer as-is.

create_plan_order_constraints() is the SAFE, manually-authored baseline: it adds the missing
cross-robot ordering by hand (robot1 doesn't even start until robot0's transfer to drop_pad is
done), exactly the kind of constraint a human engineer has to add because the automatic
extractor can't infer it. Run normally (through main_rai.py's default PlanSkeleton/dRRT*
pipeline), this environment is always safe -- dRRT*'s inter_robots_collision_fn
(rai_drrt_star.py) also independently guarantees no arm-arm collision regardless of what any
:duration says.

mm_drrt/utils/naive_duration_executor.py is what actually makes a wrong :duration visible: it
reuses this same environment and plan, but replaces dRRT*'s real inter-robot collision check
with a naive "wait out the other robot's DECLARED duration, then go" scheduler -- a common
real-world simplification. An under-estimated --transfer_duration then lets robot1 start into
drop_pad's airspace before robot0's arm has actually retreated, producing a real, detected
arm-arm collision.
"""
import itertools

import robotic as ry

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.rai_motion_planner_utils import get_placement_gen, \
    get_grasp_gen, get_gripper, arm_retrieval_motion, get_arm_motion_fn, get_trivial_roadmap, \
    get_fixed_arm_pick_place_ik_gen

from examples.rai_utils import Environment, create_box


BLOCK = (0.05, 0.05, 0.05)
DurationConflictRaiCameraSetup = [(0, -1, 4), (0, 0, 0)]

_copy_counter = itertools.count()

# Same pad convention as the other RAI envs: thin, non-collidable marker frames on top of the
# real, contact-enabled 'table' frame (pandasTable.g) -- the table itself is the only real
# collision geometry for the tabletop.
ZONE_SIZE = (0.3, 0.3, 0.02)
# Deliberately small (vs. the 0.3x0.3 zones): sample_placement()'s jitter margin (rai_utils.py)
# only leaves ~3.5cm of room per block on this footprint, so the two independently-sampled
# placements always land close together on the pad -- forcing the two arms' independently
# planned approach/retrieval paths to genuinely contest the same small volume, which is what
# lets naive_duration_executor.py's naive timeline actually observe a real arm-arm collision
# instead of the two robots just missing each other by chance.
PAD_SIZE = (0.20, 0.10, 0.02)
ZONE_X = {'left': -0.8, 'right': 0.8}
PAD_X = 0.0
SURFACE_Y = 0.15
SURFACE_Z = 0.6


class DurationConflictRaiEnvironment(Environment):
    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        super().__init__(num_objs, seed)
        if num_robots != 2 or num_objs != 2:
            raise Exception("This scenario supports exactly 2 robots and 2 objects "
                            "(one block per robot, both delivered to the shared drop_pad).")
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
        # SAFE, manually-authored baseline. Each robot delivers its own block straight to the
        # shared drop_pad -- no object links the two chains, so nothing here can be inferred
        # automatically the way a same-object handoff constraint can (see the module docstring).
        # {'pre': 'a1', 'post': 'a2'} is the constraint a human has to add by hand: robot1 may
        # not even START moving toward drop_pad until robot0's transfer there has fully finished.
        plan = {
            'a0': ('transit',  self.robots[0], self.m_objs[0], None,           self.f_objs[0]),
            'a1': ('transfer', self.robots[0], self.m_objs[0], self.f_objs[0], self.f_objs[2]),
            'a2': ('transit',  self.robots[1], self.m_objs[1], None,           self.f_objs[1]),
            'a3': ('transfer', self.robots[1], self.m_objs[1], self.f_objs[1], self.f_objs[2]),
        }
        action_orders = {self.robots[0]: ('a0', 'a1'),
                         self.robots[1]: ('a2', 'a3')}
        obj_orders = {self.m_objs[0]: ['a1'],
                      self.m_objs[1]: ['a3']}
        init_order_constraints = ({'pre': 'a1', 'post': 'a2'},)
        return plan, action_orders, obj_orders, init_order_constraints

    def create_pddl_problem(self):
        """Stock mm_drrt_manipulation PDDL 2.1 problem -- no domain changes. Tamer can solve
        this as-is; nothing in the domain prevents it from scheduling both transfers to
        drop_pad concurrently, since obj-location/robot-can-reach say nothing about two
        DIFFERENT objects sharing one fixed-obj at the same time (see module docstring)."""
        objects = {
            'robot':       [self.robots[0], self.robots[1]],
            'movable-obj': [self.m_objs[0], self.m_objs[1]],
            'fixed-obj':   [self.f_objs[0], self.f_objs[1], self.f_objs[2]],
        }
        init_state = [
            ('robot-free',         self.robots[0]),
            ('robot-free',         self.robots[1]),
            ('robot-at-base',      self.robots[0]),
            ('robot-at-base',      self.robots[1]),
            ('obj-location',       self.m_objs[0], self.f_objs[0]),
            ('obj-location',       self.m_objs[1], self.f_objs[1]),
            ('obj-clear',          self.m_objs[0]),
            ('obj-clear',          self.m_objs[1]),
            ('surface-accessible', self.f_objs[0]),
            ('surface-accessible', self.f_objs[1]),
            ('surface-accessible', self.f_objs[2]),
            ('robot-can-reach',    self.robots[0], self.f_objs[0]),
            ('robot-can-reach',    self.robots[0], self.f_objs[2]),
            ('robot-can-reach',    self.robots[1], self.f_objs[1]),
            ('robot-can-reach',    self.robots[1], self.f_objs[2]),
        ]
        goal_state = [
            ('obj-location', self.m_objs[0], self.f_objs[2]),
            ('obj-location', self.m_objs[1], self.f_objs[2]),
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
        for key, x in ZONE_X.items():
            name = f'zone_{key}'
            self._C.addFrame(name).setPosition([x, SURFACE_Y, SURFACE_Z]).setShape(ry.ST.box, size=list(ZONE_SIZE)) \
                .setColor([0.5, 0.5, 0.9]).setContact(0)
            f_objs.append(name)
        self._C.addFrame('drop_pad').setPosition([PAD_X, SURFACE_Y, SURFACE_Z]).setShape(ry.ST.box, size=list(PAD_SIZE)) \
            .setColor([0.9, 0.7, 0.3]).setContact(0)
        f_objs.append('drop_pad')

        self.robots = {0: ru.RaiRobot(self._C, ru.LEFT_ARM), 1: ru.RaiRobot(self._C, ru.RIGHT_ARM)}
        for robot in self.robots.values():
            self.custom_limits[robot] = {}
            self._grippers[robot] = get_gripper(robot)

        box_z = SURFACE_Z + ZONE_SIZE[2] / 2.0 + BLOCK[2] / 2.0
        block0_pos = (ZONE_X['left'], SURFACE_Y, box_z)
        block0_name = create_box(self._C, 'block0', block0_pos, BLOCK, color=(1.0, 0.0, 0.0))
        block1_pos = (ZONE_X['right'], SURFACE_Y, box_z)
        block1_name = create_box(self._C, 'block1', block1_pos, BLOCK, color=(0.0, 0.0, 1.0))

        self.m_objs = [block0_name, block1_name]
        self.m_objs_init_placements = {
            block0_name: ru.Pose(self._C, block0_name, (block0_pos, (1.0, 0.0, 0.0, 0.0)), 'zone_left'),
            block1_name: ru.Pose(self._C, block1_name, (block1_pos, (1.0, 0.0, 0.0, 0.0)), 'zone_right'),
        }
        self.m_obj_in_f_obj = {'zone_left': {block0_name}, 'zone_right': {block1_name}}
        self.f_objs = f_objs
