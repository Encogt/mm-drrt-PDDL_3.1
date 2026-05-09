"""
PDDL Domain Definition for MM-dRRT Manipulation Tasks

This module provides programmatic access to the MM-dRRT manipulation domain
using the Unified Planning Framework.
"""

import os
from unified_planning.shortcuts import *


def create_mm_drrt_domain():
    """
    Create MM-dRRT manipulation domain using UPF's domain builder.

    Returns:
        Domain object with robot manipulation actions (STRIPS-based)
    """
    # Define types
    Robot = UserType('robot')
    MovableObj = UserType('movable-obj')
    FixedObj = UserType('fixed-obj')

    # Define fluents (predicates)
    robot_at_base = Fluent('robot-at-base', BoolType(), r=Robot)
    robot_free = Fluent('robot-free', BoolType(), r=Robot)
    holding = Fluent('holding', BoolType(), r=Robot, m=MovableObj)
    obj_on = Fluent('obj-on', BoolType(), m=MovableObj, f=FixedObj)
    obj_clear = Fluent('obj-clear', BoolType(), m=MovableObj)
    surface_accessible = Fluent('surface-accessible', BoolType(), f=FixedObj)

    # Create actions (STRIPS-style)
    transit = InstantaneousAction('transit', r=Robot, m=MovableObj, from_f=FixedObj)
    r = transit.parameter('r')
    m = transit.parameter('m')
    from_f = transit.parameter('from_f')
    transit.add_precondition(robot_free(r))
    transit.add_precondition(obj_on(m, from_f))
    transit.add_precondition(obj_clear(m))
    transit.add_precondition(surface_accessible(from_f))
    transit.add_effect(holding(r, m), True)
    transit.add_effect(robot_free(r), False)
    transit.add_effect(obj_on(m, from_f), False)
    transit.add_effect(obj_clear(m), False)

    transfer = InstantaneousAction('transfer', r=Robot, m=MovableObj, to_f=FixedObj)
    r = transfer.parameter('r')
    m = transfer.parameter('m')
    to_f = transfer.parameter('to_f')
    transfer.add_precondition(holding(r, m))
    transfer.add_precondition(surface_accessible(to_f))
    transfer.add_effect(robot_free(r), True)
    transfer.add_effect(obj_on(m, to_f), True)
    transfer.add_effect(holding(r, m), False)
    transfer.add_effect(obj_clear(m), True)

    # Collect all fluents and actions
    fluents = [robot_at_base, robot_free, holding, obj_on, obj_clear, surface_accessible]
    actions = [transit, transfer]

    return {
        'fluents': fluents,
        'actions': actions,
        'types': {'robot': Robot, 'movable-obj': MovableObj, 'fixed-obj': FixedObj}
    }


def get_domain_pddl_string():
    """
    Return PDDL domain as string for debugging.

    Returns:
        str: PDDL domain file contents
    """
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )

    if os.path.exists(domain_file):
        with open(domain_file, 'r') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Domain file not found: {domain_file}")


def get_domain_file_path():
    """
    Get the absolute path to the PDDL domain file.

    Returns:
        str: Absolute path to domain file
    """
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../pddl/domains/mm_drrt_manipulation.pddl'
    )
    return os.path.abspath(domain_file)
