#!/usr/bin/env python
"""N Franka arms (N >= 2) around one shared drop_pad -- the scalability scenario for
demos/demo4.py.

The arms stand evenly spaced on a circle of radius ARM_RADIUS around the pad, each yawed to face
it. Every arm has its own private zone holding one block, beside its base on the outside of
the circle (away from its neighbours), and every arm has to
deliver its block onto the ONE pad all of them reach. No object links two arms' action chains, so
only a region constraint on the pad can keep them apart, and the amount of contention grows with
N: a whole-action mutex serializes N transfers; a minimal-constraint schedule only serializes the
N occupancy windows.

pandasTable.g only has two arms, so the scene file is generated: the same structure (origin
marker, contact-enabled table, `Prefix:` + `Include: panda.g` per arm, bases rigidly re-parented,
finger joints inactive), with N prefixes p0_ ... p{N-1}_.

Everything below scene construction -- subgoal sampling, compute_path, the motion-derived offset
and timing measurement -- is DurationConflictRaiEnvironment's, generalized through its
_robot_zones() hook.
"""
import math
import os
import tempfile

import robotic as ry

from mm_drrt.utils import rai_utils as ru
from mm_drrt.utils.rai_motion_planner_utils import get_gripper

from examples.rai_utils import Environment, create_box
from examples.envs.duration_conflict_rai_env import DurationConflictRaiEnvironment, BLOCK, ZONE_SIZE, \
    SURFACE_Z

RoundTableCameraSetup = [(0, -1, 4), (0, 0, 0)]

ARM_RADIUS = 0.6        # base distance from the pad centre (pad sits 0.6 m in front of each arm)
# Zone centre relative to its arm: 0.15 m BEHIND and 0.5 m to the arm's left, i.e. on the outer
# side of the circle. pandasTable.g's offset (0.45 m forward, 0.4 m left) puts a zone ~0.25 m from
# the neighbouring arm's base once N = 4 -- every pick then collides with the neighbour standing
# at carry_conf, and dRRT* never gets those arms moving (observed: subprob stuck at [4, 1, 4, 1]).
ZONE_FORWARD = -0.15
ZONE_LATERAL = 0.5
PAD_SIZE = (0.24, 0.24, 0.02)
TABLE_SIZE = 2.2
BLOCK_COLORS = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.7, 0.0), (0.9, 0.6, 0.0),
                (0.6, 0.0, 0.8), (0.0, 0.7, 0.7)]


def arm_layout(num_robots):
    """[(base_xy, yaw_degrees, zone_xy)] per arm. Arm 0 sits at the -y side, like l_panda."""
    layout = []
    for i in range(num_robots):
        theta = -math.pi / 2 + 2 * math.pi * i / num_robots
        base = (ARM_RADIUS * math.cos(theta), ARM_RADIUS * math.sin(theta))
        forward = (-math.cos(theta), -math.sin(theta))     # towards the pad
        left = (-forward[1], forward[0])
        zone = (base[0] + ZONE_FORWARD * forward[0] + ZONE_LATERAL * left[0],
                base[1] + ZONE_FORWARD * forward[1] + ZONE_LATERAL * left[1])
        layout.append((base, math.degrees(math.atan2(forward[1], forward[0])), zone))
    return layout


def write_scene_file(num_robots):
    """Generated equivalent of pandasTable.g with num_robots arms; returns its path."""
    panda = ry.raiPath('panda/panda.g')
    lines = ['world: {}',
             'origin (world): { Q: [0, 0, .6], shape: marker, size: [.03] }',
             f'table (origin): {{ Q: [0, 0, -.05], shape: ssBox, size: [{TABLE_SIZE}, {TABLE_SIZE}, .1, .02], '
             f'color: [.3, .3, .3], contact, logical:{{ }} }}']
    for i in range(num_robots):
        lines += [f'Prefix: "p{i}_"', f'Include: <{panda}>']
    lines.append('Prefix: False')
    for i, ((x, y), yaw, _) in enumerate(arm_layout(num_robots)):
        lines.append(f'Edit p{i}_panda_base (origin): {{ Q: "t({x:.4f} {y:.4f} .0) d({yaw:.4f} 0 0 1)", '
                     f'motors, joint: rigid }}')
        lines.append(f'Edit p{i}_panda_finger_joint1: {{ joint_active: False }}')
    path = os.path.join(tempfile.mkdtemp(prefix='round_table_'), f'round_table_{num_robots}.g')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# The standard carry pose. Turning it aside (joint 1 at +90 or ~149 degrees) was tried and is
# worse: sampled pick/place trajectories then hit an idle neighbour at carry on 5-10% of their
# waypoints (vs. none with this pose), which asynchronous execution cannot always schedule around.
CARRY_CONF = tuple(ru.PANDA_CARRY_CONF)


def arm_spec(i):
    return ru.ArmSpec([f'p{i}_panda_joint{j}' for j in range(1, 8)], f'p{i}_gripper', f'p{i}_',
                      CARRY_CONF)


class RoundTableRaiEnvironment(DurationConflictRaiEnvironment):
    # Leave room for the fingers between placed blocks (see DurationConflictRaiEnvironment).
    PLACEMENT_CLEARANCE = 0.03

    def __init__(self, num_robots, num_objs, arm, grasp_type, sim_id, seed):
        Environment.__init__(self, num_objs, seed)
        if num_robots < 2 or num_objs != num_robots or num_robots > len(BLOCK_COLORS):
            raise Exception(f"This scenario needs 2..{len(BLOCK_COLORS)} robots and exactly one "
                            f"block per robot (num_objs == num_robots).")
        self._num_robots = num_robots
        self._num_m_objs = num_objs
        self._C = sim_id
        self._grippers = {}
        self._grasp_type = grasp_type
        self._initialize(arm)

    @property
    def cache_name(self):
        """region_offsets_cache key: offsets measured with N arms don't carry over to another N."""
        return f'{type(self).__name__}:N{self._num_robots}'

    def _robot_zones(self):
        return [(i, self.f_objs[i]) for i in range(self._num_robots)]

    def create_plan_order_constraints(self):
        # Safe hand-written baseline, the N-arm analogue of DurationConflictRaiEnvironment's: each
        # robot only starts once the previous robot's transfer onto the pad has finished.
        plan, action_orders, obj_orders, constraints = {}, {}, {}, []
        for i in range(self._num_robots):
            transit, transfer = f'a{2 * i}', f'a{2 * i + 1}'
            plan[transit] = ('transit', self.robots[i], self.m_objs[i], None, self.f_objs[i])
            plan[transfer] = ('transfer', self.robots[i], self.m_objs[i], self.f_objs[i], self.f_objs[-1])
            action_orders[self.robots[i]] = (transit, transfer)
            obj_orders[self.m_objs[i]] = [transfer]
            if i > 0:
                constraints.append({'pre': f'a{2 * i - 1}', 'post': transit})
        return plan, action_orders, obj_orders, tuple(constraints)

    def create_pddl_problem(self):
        robots = [self.robots[i] for i in range(self._num_robots)]
        pad = self.f_objs[-1]
        objects = {'robot': robots, 'movable-obj': list(self.m_objs), 'fixed-obj': list(self.f_objs)}
        init_state = [('surface-accessible', f) for f in self.f_objs]
        goal_state = []
        for i, robot in enumerate(robots):
            init_state += [('robot-free', robot), ('robot-at-base', robot),
                           ('obj-location', self.m_objs[i], self.f_objs[i]), ('obj-clear', self.m_objs[i]),
                           ('robot-can-reach', robot, self.f_objs[i]), ('robot-can-reach', robot, pad)]
            goal_state += [('obj-location', self.m_objs[i], pad), ('robot-free', robot)]
        return objects, init_state, goal_state

    def _create_problem(self):
        self.custom_limits = {}
        n = self._num_robots
        self._C.addFile(write_scene_file(n))
        specs = [arm_spec(i) for i in range(n)]
        for spec in specs:
            ru.set_joint_positions(self._C, spec.arm_joints, spec.carry_conf)

        self.fixed_obstacles = ['table']

        box_z = SURFACE_Z + ZONE_SIZE[2] / 2.0 + BLOCK[2] / 2.0
        f_objs, m_objs = [], []
        self.m_objs_init_placements, self.m_obj_in_f_obj = {}, {}
        for i, (_, _, (zx, zy)) in enumerate(arm_layout(n)):
            zone = f'zone_{i}'
            self._C.addFrame(zone).setPosition([zx, zy, SURFACE_Z]).setShape(ry.ST.box, size=list(ZONE_SIZE)) \
                .setColor([0.5, 0.5, 0.9]).setContact(0)
            f_objs.append(zone)
            block_pos = (zx, zy, box_z)
            block = create_box(self._C, f'block{i}', block_pos, BLOCK, color=BLOCK_COLORS[i])
            m_objs.append(block)
            self.m_objs_init_placements[block] = ru.Pose(self._C, block, (block_pos, (1.0, 0.0, 0.0, 0.0)), zone)
            self.m_obj_in_f_obj[zone] = {block}
        self._C.addFrame('drop_pad').setPosition([0.0, 0.0, SURFACE_Z]).setShape(ry.ST.box, size=list(PAD_SIZE)) \
            .setColor([0.9, 0.7, 0.3]).setContact(0)
        f_objs.append('drop_pad')

        self.robots = {i: ru.RaiRobot(self._C, spec) for i, spec in enumerate(specs)}
        for robot in self.robots.values():
            self.custom_limits[robot] = {}
            self._grippers[robot] = get_gripper(robot)
        self.m_objs = m_objs
        self.f_objs = f_objs
