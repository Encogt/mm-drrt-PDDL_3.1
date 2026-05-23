"""
PDDL 3.1 Temporal Domain Definition for MM-dRRT Manipulation Tasks

Uses PDDL 3.1 object fluents (obj-location) combined with durative actions
so the planner reasons about both object location and temporal overlap between
robots. Tamer supports this combination natively.
"""

import os
from unified_planning.shortcuts import *


def create_mm_drrt_domain(transit_duration=10, transfer_duration=10):
    """
    Create MM-dRRT manipulation domain using UPF's domain builder.

    Uses PDDL 3.1 obj-location object fluent and DurativeAction so the
    planner can schedule multiple robots in parallel based on action durations.

    Returns:
        dict with keys: boolean_fluents, object_fluents, actions, types
    """
    Robot      = UserType('robot')
    MovableObj = UserType('movable-obj')
    FixedObj   = UserType('fixed-obj')

    # Boolean predicates
    robot_at_base      = Fluent('robot-at-base',       BoolType(), r=Robot)
    robot_free         = Fluent('robot-free',           BoolType(), r=Robot)
    holding            = Fluent('holding',              BoolType(), r=Robot, m=MovableObj)
    obj_clear          = Fluent('obj-clear',            BoolType(), m=MovableObj)
    surface_accessible = Fluent('surface-accessible',   BoolType(), f=FixedObj)
    robot_can_reach    = Fluent('robot-can-reach',      BoolType(), r=Robot, f=FixedObj)

    # PDDL 3.1 object fluent: returns the surface the object is on
    obj_location = Fluent('obj-location', FixedObj, m=MovableObj)

    # transit(r, m, from): pick object m from surface from
    transit = DurativeAction('transit', r=Robot, m=MovableObj, from_f=FixedObj)
    transit.set_fixed_duration(transit_duration)
    r      = transit.parameter('r')
    m      = transit.parameter('m')
    from_f = transit.parameter('from_f')
    transit.add_condition(StartTiming(), robot_free(r))
    transit.add_condition(StartTiming(), Equals(obj_location(m), from_f))
    transit.add_condition(StartTiming(), obj_clear(m))
    transit.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(from_f))
    transit.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, from_f))
    transit.add_effect(StartTiming(), robot_free(r), False)
    transit.add_effect(StartTiming(), obj_clear(m),  False)
    transit.add_effect(EndTiming(),   holding(r, m), True)

    # transfer(r, m, to): place object m on surface to
    transfer = DurativeAction('transfer', r=Robot, m=MovableObj, to_f=FixedObj)
    transfer.set_fixed_duration(transfer_duration)
    r    = transfer.parameter('r')
    m    = transfer.parameter('m')
    to_f = transfer.parameter('to_f')
    transfer.add_condition(StartTiming(), holding(r, m))
    transfer.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), surface_accessible(to_f))
    transfer.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), robot_can_reach(r, to_f))
    transfer.add_effect(EndTiming(), robot_free(r),    True)
    transfer.add_effect(EndTiming(), holding(r, m),    False)
    transfer.add_effect(EndTiming(), obj_clear(m),     True)
    transfer.add_effect(EndTiming(), obj_location(m),  to_f)

    boolean_fluents = [robot_at_base, robot_free, holding, obj_clear,
                       surface_accessible, robot_can_reach]
    object_fluents  = [obj_location]

    return {
        'boolean_fluents': boolean_fluents,
        'object_fluents':  object_fluents,
        'actions': [transit, transfer],
        'types': {'robot': Robot, 'movable-obj': MovableObj, 'fixed-obj': FixedObj}
    }


def get_domain_file_path():
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )
    return os.path.abspath(domain_file)
