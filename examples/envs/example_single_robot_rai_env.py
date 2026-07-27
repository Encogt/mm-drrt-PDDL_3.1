#!/usr/bin/env python
"""RAI port of examples/envs/example_single_robot_env.py. Same class shape/method names as the
pybullet version (see mm_drrt/utils/rai_task_planner_utils.py / rai_task_planner.py for the callers)
so PlanSkeleton needs no changes -- only the scene is built with RAI and the geometry primitives come
from mm_drrt.utils.rai_motion_planner_utils / mm_drrt.utils.rai_utils instead of pybullet."""
import itertools

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.rai_motion_planner_utils import get_ik_ir_gen, get_ik_fn, get_stable_gen, get_placement_gen, \
    base_motion, get_grasp_gen, get_gripper, get_configuration, get_arm_positions, arm_retrieval_motion, \
    get_arm_motion_fn

from examples.rai_utils import Environment, create_mobile_manipulator, create_table, create_box


BOX = (.07, .05, .15)
ExampleSingleRobotCameraSetup = [(0, -1, 4), (0, 0, 0)]

_box_counter = itertools.count()


class ExampleSingleRobotRaiEnvironment(Environment):
    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        super().__init__(num_objs, seed)
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
        start_ik_ir_fn = get_ik_ir_gen(robot, self._grippers[robot], max_attempts=25,
                                       collision_objs=start_obstacles['objs'] + self.fixed_obstacles,
                                       custom_limits=custom_limits[robot], use_debug=use_debug)
        goal_ik_ir_fn = get_ik_ir_gen(robot, self._grippers[robot], max_attempts=25,
                                      collision_objs=goal_obstacles['objs'] + self.fixed_obstacles,
                                      custom_limits=custom_limits[robot], use_debug=use_debug)
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
        if type != 'return':
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
            obj_pose = ru.Pose(self._C, m_obj)

            r_expand_configs = get_arm_positions(robot=robot, arm=self._arm)
            if type == 'transit':  # pick
                action.start['place'].assign()
                base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values, action.goal['base'].values,
                                                               obstacles=obstacles, custom_limits=self.custom_limits[robot],
                                                               num_samples=num_base_samples, expand_type='arm',
                                                               expand_configs=r_expand_configs, use_debug=use_debug)
            elif type == 'transfer':  # place
                base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values, action.goal['base'].values,
                                                               obstacles=obstacles, custom_limits=self.custom_limits[robot],
                                                               attachments=[m_obj],
                                                               num_samples=num_base_samples, expand_type='arm',
                                                               expand_configs=r_expand_configs, use_debug=use_debug)

            arm_motion_fn = get_arm_motion_fn(robot, collision_objs=obstacles, num_samples=num_arm_samples,
                                              expand_type='base', expand_configs=action.goal['base'].values, use_debug=use_debug)

            if base_roadmap:
                max_attempts = 5
                if type == 'transit':
                    attempts = 0
                    while True:
                        arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(self._arm, m_obj, action.goal['grasp'],
                                                                                         action.goal['approach_conf'], action.goal['grasp_conf'])
                        if arm_approach_roadmap is not None: break
                        attempts += 1
                        if attempts >= max_attempts: return False, [], []
                    action.goal['grasp'].attach(self._C)
                    attempts = 0
                    while True:
                        arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(
                            robot, self._arm, type, grasp=action.goal['grasp'],
                            start=arm_approach_roadmap.final_conf[len(action.goal['base'].values):],
                            goal=ru.PANDA_CARRY_CONF, obstacles=obstacles, attachments=[m_obj],
                            num_samples=num_arm_samples, expand_type='base', expand_configs=action.goal['base'].values,
                            use_debug=use_debug)
                        if arm_retrieval_roadmap is not None: break
                        attempts += 1
                        if attempts >= max_attempts: return False, [], []
                elif type == 'transfer':
                    attempts = 0
                    while True:
                        arm_approach_roadmap, arm_approach_heuristic_val = arm_motion_fn(self._arm, m_obj, action.goal['grasp'],
                                                                                         action.goal['approach_conf'], action.goal['grasp_conf'],
                                                                                         attachments=[m_obj])
                        if arm_approach_roadmap is not None: break
                        attempts += 1
                        if attempts >= max_attempts:
                            for copied_obj in copied_objs:
                                ru.remove_frame(self._C, copied_obj)
                            return False, [], []
                    # Object has been placed at its goal pose; release it from the gripper (RAI's
                    # attach() is a persistent kinematic-tree reparent, unlike pybullet's
                    # per-sample Attachment.assign(), so it must be explicitly undone here) before
                    # planning the retrieval motion, which no longer carries the object.
                    action.goal['grasp'].detach(self._C, action.to_f_obj)
                    attempts = 0
                    while True:
                        arm_retrieval_roadmap, arm_retrieval_heuristic_val = arm_retrieval_motion(
                            robot, self._arm, type, grasp=action.goal['grasp'],
                            start=arm_approach_roadmap.final_conf[len(action.goal['base'].values):],
                            goal=ru.PANDA_CARRY_CONF, obstacles=obstacles, num_samples=num_arm_samples,
                            expand_type='base', expand_configs=action.goal['base'].values, use_debug=use_debug)
                        if arm_retrieval_roadmap is not None: break
                        attempts += 1
                        if attempts >= max_attempts:
                            for copied_obj in copied_objs:
                                ru.remove_frame(self._C, copied_obj)
                            return False, [], []
                if type == 'transfer':
                    for copied_obj in copied_objs:
                        ru.remove_frame(self._C, copied_obj)
                return True, [base_roadmap, arm_approach_roadmap, arm_retrieval_roadmap], \
                       [base_heuristic_val, arm_approach_heuristic_val, arm_retrieval_heuristic_val]
            else:
                if type == 'transfer':
                    for copied_obj in copied_objs:
                        ru.remove_frame(self._C, copied_obj)
                return False, [], []
        else:  # type == 'return'
            r_expand_configs = get_arm_positions(robot=robot, arm=self._arm)
            base_roadmap, base_heuristic_val = base_motion(robot, action.start['base'].values,
                                                           action.goal['base'].values,
                                                           custom_limits=self.custom_limits[robot],
                                                           num_samples=num_base_samples, expand_type='arm',
                                                           expand_configs=r_expand_configs, use_debug=use_debug)
            if base_roadmap:
                return True, [base_roadmap], [base_heuristic_val]
            return False, [], []

    def get_joints(self, robots):
        return [ru.BASE_JOINTS + ru.ARM_JOINTS for _ in robots]

    def get_base_conf(self, robot):
        return ru.Conf(robot, ru.BASE_JOINTS, get_configuration(robot, 'base'))

    def get_init_base_conf(self, robot):
        return ru.Conf(robot, ru.BASE_JOINTS, self.robots_init_poses[robot])

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
        plan = {'a0': ('transit', self.robots[0], self.m_objs[0], None, self.f_objs[0]),
                'a1': ('transfer', self.robots[0], self.m_objs[0], self.f_objs[0], self.f_objs[1])
                }
        action_orders = {self.robots[0]: ('a0', 'a1')}
        obj_orders = {self.m_objs[0]: ['a1']}
        init_order_constraints = ()
        return plan, action_orders, obj_orders, init_order_constraints

    def _create_problem(self):
        self.custom_limits = {}

        # Desk-height table (not a 1.2m pedestal): the taller table originally ported over from the
        # pybullet scene left too little vertical clearance for the Panda's palm/fingers to reach a
        # box sitting on top of it without colliding with the table itself (see plan verification
        # notes). TABLE_H / TABLE_SIZE / box_z are kept in sync with BOX's placement below.
        TABLE_SIZE = (0.4, 0.4, 0.75)
        table = [create_table(self._C, 'table0', (2, 0, TABLE_SIZE[2] / 2), size=TABLE_SIZE),
                create_table(self._C, 'table1', (-2, 0, TABLE_SIZE[2] / 2), size=TABLE_SIZE)]
        self.fixed_obstacles = list(table)

        box_z = TABLE_SIZE[2] + BOX[2] / 2
        boxes = []
        self.m_objs_init_placements = {}
        for i in range(self._num_m_objs):
            if i == 0:
                name = create_box(self._C, 'box0', (2.05, 0.1, box_z), BOX)
                boxes.append(name)
            else:
                raise Exception("The number of objects is more than one.")
            box_pose = ((2.05, 0.1, box_z), (1.0, 0.0, 0.0, 0.0))
            self.m_objs_init_placements[boxes[i]] = ru.Pose(self._C, boxes[i], box_pose, table[0])

        self.robots = {}
        self.robots_init_poses = {}
        rotate_z = 0.5
        for i in range(self._num_robots):
            if i == 0:
                self.robots[i] = create_mobile_manipulator(self._C, robot=i, base_pose=(0, 0, rotate_z * i))
                self.robots_init_poses[self.robots[i]] = (0, 0, rotate_z * i)
                self.custom_limits[self.robots[i]] = {'ranger_transX': (-3, 3), 'ranger_transY': (-3, 3)}
            else:
                raise Exception("The number of robots is more than one.")

        for robot in self.robots.values():
            self._grippers[robot] = get_gripper(robot)

        self.m_objs = boxes
        self.f_objs = table
        self.m_obj_in_f_obj = {table[0]: set(boxes)}
