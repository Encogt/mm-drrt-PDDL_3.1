"""
PDDL 3.1 Domain Definition for MM-dRRT Manipulation Tasks

Uses object fluents (:object-fluents) to represent object locations,
replacing the boolean obj-on predicate with the obj-location function.
"""

import os
from unified_planning.shortcuts import *


def create_mm_drrt_domain():
    """
    Create MM-dRRT manipulation domain using UPF's domain builder.

    Uses PDDL 3.1 object fluents: obj-location(m) returns the fixed-obj
    surface the movable object currently rests on.

    Returns:
        dict with keys: boolean_fluents, object_fluents, actions, types
    """
    Robot = UserType('robot')
    MovableObj = UserType('movable-obj')
    FixedObj = UserType('fixed-obj')

    # Boolean predicates
    robot_at_base = Fluent('robot-at-base', BoolType(), r=Robot)
    robot_free = Fluent('robot-free', BoolType(), r=Robot)
    holding = Fluent('holding', BoolType(), r=Robot, m=MovableObj)
    obj_clear = Fluent('obj-clear', BoolType(), m=MovableObj)
    surface_accessible = Fluent('surface-accessible', BoolType(), f=FixedObj)
    robot_can_reach = Fluent('robot-can-reach', BoolType(), r=Robot, f=FixedObj)

    # PDDL 3.1 object fluent: returns the surface the object is on
    obj_location = Fluent('obj-location', FixedObj, m=MovableObj)

    # transit(r, m, from): pick object m from surface from
    transit = InstantaneousAction('transit', r=Robot, m=MovableObj, from_f=FixedObj)
    r = transit.parameter('r')
    m = transit.parameter('m')
    from_f = transit.parameter('from_f')
    transit.add_precondition(robot_free(r))
    transit.add_precondition(Equals(obj_location(m), from_f))
    transit.add_precondition(obj_clear(m))
    transit.add_precondition(surface_accessible(from_f))
    transit.add_precondition(robot_can_reach(r, from_f))
    transit.add_effect(holding(r, m), True)
    transit.add_effect(robot_free(r), False)
    transit.add_effect(obj_clear(m), False)

    # transfer(r, m, to): place object m on surface to
    transfer = InstantaneousAction('transfer', r=Robot, m=MovableObj, to_f=FixedObj)
    r = transfer.parameter('r')
    m = transfer.parameter('m')
    to_f = transfer.parameter('to_f')
    transfer.add_precondition(holding(r, m))
    transfer.add_precondition(surface_accessible(to_f))
    transfer.add_precondition(robot_can_reach(r, to_f))
    transfer.add_effect(robot_free(r), True)
    transfer.add_effect(holding(r, m), False)
    transfer.add_effect(obj_clear(m), True)
    transfer.add_effect(obj_location(m), to_f)

    boolean_fluents = [robot_at_base, robot_free, holding, obj_clear, surface_accessible, robot_can_reach]
    object_fluents = [obj_location]

    return {
        'boolean_fluents': boolean_fluents,
        'object_fluents': object_fluents,
        'actions': [transit, transfer],
        'types': {'robot': Robot, 'movable-obj': MovableObj, 'fixed-obj': FixedObj}
    }


def get_domain_pddl_string():
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )
    if os.path.exists(domain_file):
        with open(domain_file, 'r') as f:
            return f.read()
    raise FileNotFoundError(f"Domain file not found: {domain_file}")


def get_domain_file_path():
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )
    return os.path.abspath(domain_file)
