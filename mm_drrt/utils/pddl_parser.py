"""
PDDL Plan Parser for MM-dRRT

This module converts PDDL plans (from UPF or other planners) into MM-dRRT's
expected format: (plan, action_orders, obj_orders, init_order_constraints).
"""

from collections import defaultdict


def parse_pddl_plan(pddl_plan, mapper, env):
    """
    Convert PDDL plan to MM-dRRT format.

    Args:
        pddl_plan: Plan from UPF planner (SequentialPlan or TimeTriggeredPlan)
        mapper: ObjectMapper for converting PDDL names to PyBullet IDs
        env: Environment instance for validation

    Returns:
        tuple: (plan, action_orders, obj_orders, init_order_constraints)
            plan: dict mapping action_name → (type, robot, obj, from, to)
            action_orders: dict mapping robot → tuple of action names
            obj_orders: dict mapping obj → list of actions manipulating it
            init_order_constraints: tuple of {'pre': action_name, 'post': action_name}
    """
    plan = {}
    action_orders = defaultdict(list)
    obj_orders = defaultdict(list)

    # Parse actions from PDDL plan
    actions_list = []

    if hasattr(pddl_plan, 'timed_actions'):
        # TimeTriggeredPlan (temporal/durative actions)
        for i, (start_time, action, duration) in enumerate(pddl_plan.timed_actions):
            action_name = f"a{i}"
            action_info = _parse_action(action, action_name, mapper, start_time, start_time + duration)
            actions_list.append(action_info)
            plan[action_name] = action_info['mm_drrt_format']
    else:
        # SequentialPlan (non-temporal actions)
        for i, action in enumerate(pddl_plan.actions):
            action_name = f"a{i}"
            action_info = _parse_action(action, action_name, mapper, i, i + 1)
            actions_list.append(action_info)
            plan[action_name] = action_info['mm_drrt_format']

    # Extract action orders (group by robot)
    robot_actions = defaultdict(list)
    for action_info in actions_list:
        robot = action_info['robot']
        action_name = action_info['name']
        robot_actions[robot].append((action_info['start_time'], action_name))

    # Sort by start time and create action_orders
    for robot, actions in robot_actions.items():
        sorted_actions = sorted(actions, key=lambda x: x[0])
        action_orders[robot] = tuple(action[1] for action in sorted_actions)

    # Extract object orders (track which actions manipulate which objects)
    for action_info in actions_list:
        if action_info['type'] == 'transfer':
            # Transfer actions place objects
            obj = action_info['movable_obj']
            action_name = action_info['name']
            obj_orders[obj].append(action_name)

    # Extract temporal constraints (for multi-robot coordination)
    init_order_constraints = extract_temporal_constraints(actions_list)

    return plan, dict(action_orders), dict(obj_orders), init_order_constraints


def _parse_action(action, action_name, mapper, start_time, end_time):
    """
    Parse a single PDDL action into MM-dRRT format.

    Args:
        action: UPF action instance
        action_name: Assigned action name (e.g., 'a0')
        mapper: ObjectMapper
        start_time: Action start time
        end_time: Action end time

    Returns:
        dict: Action information including MM-dRRT format
    """
    action_type = action.action.name
    parameters = action.actual_parameters

    # Extract robot from parameters
    robot_pddl = parameters[0].object().name
    robot = mapper.get_pybullet_obj(robot_pddl)

    if action_type == 'transit':
        # transit(robot, movable-obj, from-fixed-obj)
        movable_obj_pddl = parameters[1].object().name
        from_fixed_obj_pddl = parameters[2].object().name

        movable_obj = mapper.get_pybullet_obj(movable_obj_pddl)
        from_fixed_obj = mapper.get_pybullet_obj(from_fixed_obj_pddl)

        # MM-dRRT format for transit: ('transit', robot, m_obj, None, from_f_obj)
        mm_drrt_format = ('transit', robot, movable_obj, None, from_fixed_obj)

        return {
            'name': action_name,
            'type': 'transit',
            'robot': robot,
            'movable_obj': movable_obj,
            'from_fixed_obj': from_fixed_obj,
            'to_fixed_obj': from_fixed_obj,  # For MM-dRRT, this is the target for IK
            'mm_drrt_format': mm_drrt_format,
            'start_time': start_time,
            'end_time': end_time
        }

    elif action_type == 'transfer':
        # transfer(robot, movable-obj, to-fixed-obj)
        movable_obj_pddl = parameters[1].object().name
        to_fixed_obj_pddl = parameters[2].object().name

        movable_obj = mapper.get_pybullet_obj(movable_obj_pddl)
        to_fixed_obj = mapper.get_pybullet_obj(to_fixed_obj_pddl)

        # Need to infer from_fixed_obj (where it was picked from)
        # This should be tracked from the preceding transit action
        # For now, we'll set it to None and let MM-dRRT's refinement handle it
        from_fixed_obj = None

        # MM-dRRT format for transfer: ('transfer', robot, m_obj, from_f_obj, to_f_obj)
        mm_drrt_format = ('transfer', robot, movable_obj, from_fixed_obj, to_fixed_obj)

        return {
            'name': action_name,
            'type': 'transfer',
            'robot': robot,
            'movable_obj': movable_obj,
            'from_fixed_obj': from_fixed_obj,
            'to_fixed_obj': to_fixed_obj,
            'mm_drrt_format': mm_drrt_format,
            'start_time': start_time,
            'end_time': end_time
        }

    elif action_type == 'return-to-base':
        # return-to-base(robot)
        mm_drrt_format = ('return', robot, None, None, None)

        return {
            'name': action_name,
            'type': 'return',
            'robot': robot,
            'movable_obj': None,
            'from_fixed_obj': None,
            'to_fixed_obj': None,
            'mm_drrt_format': mm_drrt_format,
            'start_time': start_time,
            'end_time': end_time
        }

    else:
        raise ValueError(f"Unknown action type: {action_type}")


def _infer_from_fixed_obj(actions_list):
    """
    Infer from_fixed_obj for transfer actions by looking at preceding transit actions.

    Args:
        actions_list: List of parsed action info dicts

    Returns:
        None (modifies actions_list in place)
    """
    # Build a map of robot → last transit action
    robot_last_transit = {}

    for action_info in actions_list:
        robot = action_info['robot']

        if action_info['type'] == 'transit':
            robot_last_transit[robot] = action_info

        elif action_info['type'] == 'transfer':
            # Update from_fixed_obj based on preceding transit
            if robot in robot_last_transit:
                transit_info = robot_last_transit[robot]
                from_fixed = transit_info['from_fixed_obj']

                # Update the transfer action's from_fixed_obj
                action_info['from_fixed_obj'] = from_fixed

                # Update mm_drrt_format tuple
                mm_drrt_format = list(action_info['mm_drrt_format'])
                mm_drrt_format[3] = from_fixed
                action_info['mm_drrt_format'] = tuple(mm_drrt_format)


def extract_action_orders(actions_list):
    """
    Build robot → action sequence mapping.

    Args:
        actions_list: List of parsed action info dicts

    Returns:
        dict: Robot → tuple of action names
    """
    robot_actions = defaultdict(list)

    for action_info in actions_list:
        robot = action_info['robot']
        action_name = action_info['name']
        start_time = action_info['start_time']
        robot_actions[robot].append((start_time, action_name))

    # Sort by start time
    action_orders = {}
    for robot, actions in robot_actions.items():
        sorted_actions = sorted(actions, key=lambda x: x[0])
        action_orders[robot] = tuple(action[1] for action in sorted_actions)

    return action_orders


def extract_obj_orders(actions_list):
    """
    Build object → action list mapping.

    Args:
        actions_list: List of parsed action info dicts

    Returns:
        dict: Object → list of actions manipulating it
    """
    obj_orders = defaultdict(list)

    for action_info in actions_list:
        if action_info['type'] in ['transit', 'transfer']:
            obj = action_info['movable_obj']
            action_name = action_info['name']

            # Only track transfer actions for obj_orders (placement actions)
            if action_info['type'] == 'transfer':
                obj_orders[obj].append(action_name)

    return dict(obj_orders)


def extract_temporal_constraints(actions_list):
    """
    Extract precedence constraints from temporal plan.

    For each pair of actions from different robots:
    - If action_i.end <= action_j.start: add constraint

    Args:
        actions_list: List of parsed action info dicts

    Returns:
        tuple: Temporal constraints in MM-dRRT format
    """
    constraints = []

    # Sort by start time
    sorted_actions = sorted(actions_list, key=lambda a: a['start_time'])

    for i, action_i in enumerate(sorted_actions):
        for action_j in sorted_actions[i + 1:]:
            # Only add constraints between different robots
            if action_i['robot'] != action_j['robot']:
                # If action_i completes before action_j starts
                if action_i['end_time'] <= action_j['start_time']:
                    constraints.append({
                        'pre': action_i['name'],
                        'post': action_j['name']
                    })

    return tuple(constraints)


def validate_plan(plan, action_orders, env):
    """
    Validate that the parsed plan is consistent with the environment.

    Args:
        plan: MM-dRRT plan dict
        action_orders: Robot → action sequence mapping
        env: Environment instance

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValueError: If plan is invalid
    """
    # Check that all robots have action sequences
    if hasattr(env, 'robots'):
        for robot in env.robots:
            if robot not in action_orders:
                raise ValueError(f"Robot {robot} missing from action_orders")

    # Check that all actions in action_orders exist in plan
    for robot, actions in action_orders.items():
        for action_name in actions:
            if action_name not in plan:
                raise ValueError(f"Action {action_name} in action_orders but not in plan")

    return True
