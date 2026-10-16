"""RAI scene-building helpers, paralleling examples/utils.py's pybullet version."""
import abc

import robotic as ry

from mm_drrt.utils import rai_utils as ru


def create_mobile_manipulator(C, robot=0, base_pose=(0.0, 0.0, 0.0)):
    """Loads RAI's built-in Panda-on-Ranger scenario into C and returns the robot id (there is only
    one shared ry.Config per scene in this POC, so the id is just a label, not a separate body)."""
    if robot != 0:
        raise Exception("The RAI POC only supports a single robot.")
    C.addFile(ry.raiPath(ru.ROBOT_SCENARIO))
    ru.set_joint_positions(C, ru.BASE_JOINTS, base_pose)
    ru.set_joint_positions(C, ru.ARM_JOINTS, ru.PANDA_CARRY_CONF)
    return C


def create_table(C, name, position, size=(0.4, 0.4, 1.2), color=(1.0, 1.0, 1.0)):
    C.addFrame(name).setPosition(list(position)).setShape(ry.ST.box, size=list(size)) \
        .setColor(list(color)).setContact(1)
    return name


def create_box(C, name, position, size, color=(1.0, 0.0, 0.0)):
    C.addFrame(name).setPosition(list(position)).setShape(ry.ST.box, size=list(size)) \
        .setColor(list(color)).setContact(1)
    return name


class Environment:
    """Base class for an environment. Identical to examples.utils.Environment (no pybullet in it)."""
    def __init__(self, num_objs, seed):
        self._num_objs = num_objs
        self._seed = seed

    @abc.abstractmethod
    def create_plan_order_constraints(self):
        raise NotImplementedError("Override me!")
